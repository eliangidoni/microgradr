use std::{
    cell::{Ref, RefCell, RefMut},
    collections::HashSet,
    f64::consts::LN_10,
    fmt,
    iter::Sum,
    ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub},
    rc::Rc,
};

#[derive(Clone, Eq, PartialEq, Debug, PartialOrd)]
pub struct Value {
    inner: ValueInnerRef,
}

type ValueInnerRef = Rc<RefCell<ValueInner>>;

type BackwardFun = fn(Ref<ValueInner>, &mut Vec<f64>);

struct ValueInner {
    name: String,
    data: f64,
    grad: f64,
    grad_idx: usize,
    left: Option<Value>,
    right: Option<Value>,
    backward_fn: Option<BackwardFun>,
    uuid: uuid::Uuid,
    op: String,
}

impl Value {
    pub fn with_name(&self, name: String) -> Self {
        self.borrow_mut().name = name;
        self.clone()
    }

    pub fn data(&self) -> f64 {
        self.borrow().data
    }

    pub fn update(&mut self, rate: f64) {
        self.borrow_mut().data += rate * self.grad();
    }

    pub fn zero_grad(&mut self) {
        self.borrow_mut().grad = 0.0;
    }

    pub fn grad(&self) -> f64 {
        self.borrow().grad
    }

    pub fn topological_order(
        node: &Self,
        visited: &mut HashSet<uuid::Uuid>,
        path: &mut Vec<Self>,
    ) -> () {
        if visited.contains(&node.borrow().uuid) {
            return;
        }
        visited.insert(node.borrow().uuid);
        if let Some(left) = &node.borrow().left {
            Self::topological_order(left, visited, path);
        }
        if let Some(right) = &node.borrow().right {
            Self::topological_order(right, visited, path);
        }
        path.push(node.clone());
    }

    pub fn backward(&self) {
        let mut grads = Vec::<f64>::new();
        let mut visited = HashSet::<uuid::Uuid>::new();
        let mut path = Vec::<Self>::new();
        Self::topological_order(self, &mut visited, &mut path);
        for (i, node) in path.iter().enumerate() {
            node.borrow_mut().grad_idx = i;
            grads.push(0.0);
        }
        grads[self.borrow().grad_idx] = 1.0;
        for node in path.iter().rev() {
            let inner = node.borrow();
            if let Some(backward_fn) = inner.backward_fn {
                backward_fn(inner, &mut grads);
            }
        }
        for node in path.iter() {
            let mut inner = node.borrow_mut();
            inner.grad += grads[inner.grad_idx];
        }
    }

    pub fn pow(&self, other: &Self) -> Self {
        if other.borrow().left.is_some() || other.borrow().right.is_some() {
            panic!("The exponent must be a number");
        }
        let backward_fn = |val: Ref<ValueInner>, grads: &mut Vec<f64>| {
            let left = val.left.as_ref().unwrap().borrow();
            let right = val.right.as_ref().unwrap().borrow();
            grads[left.grad_idx] +=
                (right.data * left.data.powf(right.data - 1.0)) * grads[val.grad_idx];
        };
        Self::with_children(
            self.data().powf(other.data()),
            (Some(self.clone()), Some(other.clone())),
            Some(backward_fn),
            "pow".to_string(),
        )
    }

    pub fn relu(&self) -> Self {
        let backward_fn = |val: Ref<ValueInner>, grads: &mut Vec<f64>| {
            let left = val.left.as_ref().unwrap().borrow();
            grads[left.grad_idx] += (val.data > 0.0) as i32 as f64 * grads[val.grad_idx];
        };
        Self::with_children(
            self.data().max(0.0),
            (Some(self.clone()), None),
            Some(backward_fn),
            "relu".to_string(),
        )
    }

    pub fn tanh(&self) -> Self {
        let backward_fn = |val: Ref<ValueInner>, grads: &mut Vec<f64>| {
            let left = val.left.as_ref().unwrap().borrow();
            grads[left.grad_idx] += (1.0 - val.data.powf(2.0)) * grads[val.grad_idx];
        };
        Self::with_children(
            self.data().tanh(),
            (Some(self.clone()), None),
            Some(backward_fn),
            "tanh".to_string(),
        )
    }

    pub fn exp(&self) -> Self {
        let backward_fn = |val: Ref<ValueInner>, grads: &mut Vec<f64>| {
            let left = val.left.as_ref().unwrap().borrow();
            grads[left.grad_idx] += left.data.exp() * grads[val.grad_idx];
        };
        Self::with_children(
            self.data().exp(),
            (Some(self.clone()), None),
            Some(backward_fn),
            "exp".to_string(),
        )
    }

    pub fn log10(&self) -> Self {
        let backward_fn = |val: Ref<ValueInner>, grads: &mut Vec<f64>| {
            let left = val.left.as_ref().unwrap().borrow();
            grads[left.grad_idx] += (1.0 / (left.data * LN_10)) * grads[val.grad_idx];
        };
        Self::with_children(
            self.data().log10(),
            (Some(self.clone()), None),
            Some(backward_fn),
            "log10".to_string(),
        )
    }

    pub fn log(&self) -> Self {
        let backward_fn = |val: Ref<ValueInner>, grads: &mut Vec<f64>| {
            let left = val.left.as_ref().unwrap().borrow();
            grads[left.grad_idx] += (1.0 / left.data) * grads[val.grad_idx];
        };
        Self::with_children(
            self.data().ln(),
            (Some(self.clone()), None),
            Some(backward_fn),
            "logn".to_string(),
        )
    }

    pub fn sigmoid(&self) -> Self {
        let backward_fn = |val: Ref<ValueInner>, grads: &mut Vec<f64>| {
            let left = val.left.as_ref().unwrap().borrow();
            grads[left.grad_idx] += val.data * (1.0 - val.data) * grads[val.grad_idx];
        };
        Self::with_children(
            1.0 / (1.0 + (-self.data()).exp()),
            (Some(self.clone()), None),
            Some(backward_fn),
            "sigmoid".to_string(),
        )
    }

    fn new(v: ValueInner) -> Self {
        Self {
            inner: Rc::new(RefCell::new(v)),
        }
    }

    fn with_children(
        data: f64,
        children: (Option<Self>, Option<Self>),
        backward_fn: Option<BackwardFun>,
        op: String,
    ) -> Self {
        Self::new(ValueInner {
            uuid: uuid::Uuid::new_v4(),
            name: "".to_string(),
            data,
            grad: 0.0,
            grad_idx: 0,
            left: children.0,
            right: children.1,
            backward_fn: backward_fn,
            op: op,
        })
    }

    fn add_impl(&self, other: &Self) -> Self {
        let backward_fn = |val: Ref<ValueInner>, grads: &mut Vec<f64>| {
            let left = val.left.as_ref().unwrap().borrow();
            let right = val.right.as_ref().unwrap().borrow();
            grads[left.grad_idx] += grads[val.grad_idx];
            grads[right.grad_idx] += grads[val.grad_idx];
        };
        Self::with_children(
            self.data() + other.data(),
            (Some(self.clone()), Some(other.clone())),
            Some(backward_fn),
            "+".to_string(),
        )
    }

    fn mul_impl(&self, other: &Self) -> Self {
        let backward_fn = |val: Ref<ValueInner>, grads: &mut Vec<f64>| {
            let left = val.left.as_ref().unwrap().borrow();
            let right = val.right.as_ref().unwrap().borrow();
            grads[left.grad_idx] += right.data * grads[val.grad_idx];
            grads[right.grad_idx] += left.data * grads[val.grad_idx];
        };
        Self::with_children(
            self.data() * other.data(),
            (Some(self.clone()), Some(other.clone())),
            Some(backward_fn),
            "*".to_string(),
        )
    }

    fn borrow(&self) -> Ref<ValueInner> {
        self.inner.borrow()
    }

    fn borrow_mut(&self) -> RefMut<ValueInner> {
        self.inner.borrow_mut()
    }
}

impl From<f64> for Value {
    fn from(v: f64) -> Value {
        Value::new(ValueInner {
            uuid: uuid::Uuid::new_v4(),
            name: "".to_string(),
            data: v,
            grad: 0.0,
            grad_idx: 0,
            left: None,
            right: None,
            backward_fn: None,
            op: "".to_string(),
        })
    }
}

impl From<usize> for Value {
    fn from(v: usize) -> Value {
        Value::new(ValueInner {
            uuid: uuid::Uuid::new_v4(),
            name: "".to_string(),
            data: v as f64,
            grad: 0.0,
            grad_idx: 0,
            left: None,
            right: None,
            backward_fn: None,
            op: "".to_string(),
        })
    }
}

impl From<bool> for Value {
    fn from(v: bool) -> Value {
        Value::new(ValueInner {
            uuid: uuid::Uuid::new_v4(),
            name: "".to_string(),
            data: v as usize as f64,
            grad: 0.0,
            grad_idx: 0,
            left: None,
            right: None,
            backward_fn: None,
            op: "".to_string(),
        })
    }
}

impl Add<&Value> for Value {
    type Output = Value;

    fn add(self, other: &Value) -> Value {
        self.add_impl(other)
    }
}

impl Add for Value {
    type Output = Value;

    fn add(self, other: Value) -> Value {
        self.add_impl(&other)
    }
}

impl<'a, 'b> Add<&'a Value> for &'b Value {
    type Output = Value;

    fn add(self, other: &'a Value) -> Value {
        self.add_impl(other)
    }
}

impl<'a> Add<Value> for &'a Value {
    type Output = Value;

    fn add(self, other: Value) -> Value {
        self.add_impl(&other)
    }
}

impl Sub<&Value> for Value {
    type Output = Value;

    fn sub(self, other: &Value) -> Value {
        self + (-other)
    }
}

impl Sub for Value {
    type Output = Value;

    fn sub(self, other: Value) -> Value {
        self + (-other)
    }
}

impl<'a, 'b> Sub<&'a Value> for &'b Value {
    type Output = Value;

    fn sub(self, other: &'a Value) -> Value {
        self + (-other)
    }
}

impl<'a> Sub<Value> for &'a Value {
    type Output = Value;

    fn sub(self, other: Value) -> Value {
        self + (-other)
    }
}

impl Mul<&Value> for Value {
    type Output = Value;

    fn mul(self, other: &Value) -> Value {
        self.mul_impl(other)
    }
}

impl Mul for Value {
    type Output = Value;

    fn mul(self, other: Value) -> Value {
        self.mul_impl(&other)
    }
}

impl<'a, 'b> Mul<&'a Value> for &'b Value {
    type Output = Value;

    fn mul(self, other: &'a Value) -> Value {
        self.mul_impl(other)
    }
}

impl<'a> Mul<Value> for &'a Value {
    type Output = Value;

    fn mul(self, other: Value) -> Value {
        self.mul_impl(&other)
    }
}

impl Div for Value {
    type Output = Value;

    fn div(self, other: Value) -> Value {
        if other.data() == 0.0 {
            panic!("Division by zero");
        }
        self * other.pow(&Value::from(-1.0))
    }
}

impl Div<&Value> for Value {
    type Output = Value;

    fn div(self, other: &Value) -> Value {
        if other.data() == 0.0 {
            panic!("Division by zero");
        }
        self * other.pow(&Value::from(-1.0))
    }
}

impl<'a, 'b> Div<&'a Value> for &'b Value {
    type Output = Value;

    fn div(self, other: &'a Value) -> Value {
        if other.data() == 0.0 {
            panic!("Division by zero");
        }
        self * other.pow(&Value::from(-1.0))
    }
}

impl<'a> Div<Value> for &'a Value {
    type Output = Value;

    fn div(self, other: Value) -> Value {
        if other.data() == 0.0 {
            panic!("Division by zero");
        }
        self * other.pow(&Value::from(-1.0))
    }
}

impl Neg for Value {
    type Output = Value;

    fn neg(self) -> Value {
        self * Value::from(-1.0)
    }
}

impl<'a> Neg for &'a Value {
    type Output = Value;

    fn neg(self) -> Value {
        self * Value::from(-1.0)
    }
}

impl Sum for Value {
    fn sum<I>(iter: I) -> Value
    where
        I: Iterator<Item = Value>,
    {
        iter.fold(Value::from(0.0), |acc, x| acc + x)
    }
}

impl<'a> Sum<&'a Value> for Value {
    fn sum<I>(iter: I) -> Value
    where
        I: Iterator<Item = &'a Value>,
    {
        iter.fold(Value::from(0.0), |acc, x| acc + x)
    }
}

impl fmt::Debug for ValueInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ValueInner")
            .field("op", &self.op)
            .field("name", &self.name)
            .field("data", &self.data)
            .field("grad", &self.grad)
            .finish()
    }
}

impl PartialEq for ValueInner {
    fn eq(&self, other: &Self) -> bool {
        self.data.eq(&other.data)
    }
}

impl Eq for ValueInner {}

impl PartialOrd for ValueInner {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.data.partial_cmp(&other.data)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd)]
pub struct ValueVec {
    values: Vec<Value>,
}

impl ValueVec {
    pub fn new() -> Self {
        Self {
            values: Vec::<Value>::new(),
        }
    }

    pub fn data(&self) -> Vec<f64> {
        self.values.iter().map(|v| v.data()).collect()
    }

    pub fn grad(&self) -> Vec<f64> {
        self.values.iter().map(|v| v.grad()).collect()
    }

    pub fn update(&mut self, rate: f64) {
        self.values.iter_mut().for_each(|v| v.update(rate));
    }

    pub fn zero_grad(&mut self) {
        self.values.iter_mut().for_each(|v| v.zero_grad());
    }

    pub fn append(&mut self, other: &mut Self) {
        self.values.append(&mut other.values);
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn clear(&mut self) {
        self.values.clear();
    }

    pub fn insert(&mut self, index: usize, value: Value) {
        self.values.insert(index, value);
    }

    pub fn remove(&mut self, index: usize) -> Value {
        self.values.remove(index)
    }

    pub fn pop(&mut self) -> Option<Value> {
        self.values.pop()
    }

    pub fn push(&mut self, value: Value) {
        self.values.push(value);
    }

    pub fn iter(&self) -> ValueVecIter {
        ValueVecIter {
            values: self,
            index: 0,
        }
    }

    pub fn backward(&self) {
        self.iter().for_each(|v| v.backward());
    }

    pub fn relu(&self) -> Self {
        let mut res = Self::new();
        for val in self.values.iter() {
            res.push(val.relu());
        }
        res
    }

    pub fn tanh(&self) -> Self {
        let mut res = Self::new();
        for val in self.values.iter() {
            res.push(val.tanh());
        }
        res
    }

    pub fn exp(&self) -> Self {
        let mut res = Self::new();
        for val in self.values.iter() {
            res.push(val.exp());
        }
        res
    }

    pub fn log10(&self) -> Self {
        let mut res = Self::new();
        for val in self.values.iter() {
            res.push(val.log10());
        }
        res
    }

    pub fn log(&self) -> Self {
        let mut res = Self::new();
        for val in self.values.iter() {
            res.push(val.log());
        }
        res
    }

    pub fn sigmoid(&self) -> Self {
        let mut res = Self::new();
        for val in self.values.iter() {
            res.push(val.sigmoid());
        }
        res
    }

    pub fn sum(&self) -> Value {
        self.values.iter().sum::<Value>()
    }

    pub fn mean(&self) -> Value {
        assert!(!self.is_empty());
        self.sum() / Value::from(self.len())
    }

    pub fn max_index(&self) -> (Value, Value) {
        assert!(!self.is_empty());
        let mut max = Option::<&Value>::None;
        let mut maxidx = 0;
        for (idx, val) in self.iter().enumerate() {
            match max {
                Some(v) if val.data() > v.data() => (max, maxidx) = (Some(val), idx),
                Some(_) => (),
                None => (max, maxidx) = (Some(val), idx),
            }
        }
        (max.unwrap().clone(), Value::from(maxidx))
    }

    pub fn max(&self) -> Value {
        self.max_index().0
    }

    pub fn min_index(&self) -> (Value, Value) {
        assert!(!self.is_empty());
        let mut min = Option::<&Value>::None;
        let mut minidx = 0;
        for (idx, val) in self.iter().enumerate() {
            match min {
                Some(v) if val.data() < v.data() => (min, minidx) = (Some(val), idx),
                Some(_) => (),
                None => (min, minidx) = (Some(val), idx),
            }
        }
        (min.unwrap().clone(), Value::from(minidx))
    }

    pub fn min(&self) -> Value {
        self.min_index().0
    }

    pub fn softmax(&self) -> Self {
        assert!(!self.is_empty());
        let exps = self.exp();
        let ref sum = exps.values.iter().sum::<Value>();
        let mut res = Self::new();
        for exp in exps {
            res.push(exp / sum);
        }
        res
    }

    pub fn pow(&self, other: &Self) -> Self {
        assert_eq!(self.len(), other.len());
        let mut res = Self::new();
        for (a, b) in self.values.iter().zip(other.values.iter()) {
            res.push(a.pow(b));
        }
        res
    }

    pub fn add_impl(&self, other: &Self) -> Self {
        assert_eq!(self.len(), other.len());
        let mut res = Self::new();
        for (a, b) in self.values.iter().zip(other.values.iter()) {
            res.push(a.add_impl(b));
        }
        res
    }

    pub fn mul_impl(&self, other: &Self) -> Self {
        assert_eq!(self.len(), other.len());
        let mut res = Self::new();
        for (a, b) in self.values.iter().zip(other.values.iter()) {
            res.push(a.mul_impl(b));
        }
        res
    }
}

impl From<Vec<f64>> for ValueVec {
    fn from(other: Vec<f64>) -> ValueVec {
        let mut v = ValueVec::new();
        other
            .into_iter()
            .for_each(|val| v.values.push(Value::from(val)));
        v
    }
}

impl Index<usize> for ValueVec {
    type Output = Value;

    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index]
    }
}

impl IndexMut<usize> for ValueVec {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.values[index]
    }
}

impl Add<&ValueVec> for ValueVec {
    type Output = ValueVec;

    fn add(self, other: &ValueVec) -> ValueVec {
        self.add_impl(other)
    }
}

impl Add for ValueVec {
    type Output = ValueVec;

    fn add(self, other: ValueVec) -> ValueVec {
        self.add_impl(&other)
    }
}

impl<'a, 'b> Add<&'a ValueVec> for &'b ValueVec {
    type Output = ValueVec;

    fn add(self, other: &'a ValueVec) -> ValueVec {
        self.add_impl(other)
    }
}

impl<'a> Add<ValueVec> for &'a ValueVec {
    type Output = ValueVec;

    fn add(self, other: ValueVec) -> ValueVec {
        self.add_impl(&other)
    }
}

impl Add<Value> for ValueVec {
    type Output = ValueVec;

    fn add(self, other: Value) -> ValueVec {
        self.add_impl(&ValueVec::from(vec![other.data(); self.len()]))
    }
}

impl Add<ValueVec> for Value {
    type Output = ValueVec;

    fn add(self, other: ValueVec) -> ValueVec {
        other.add_impl(&ValueVec::from(vec![self.data(); other.len()]))
    }
}

impl Add<&Value> for ValueVec {
    type Output = ValueVec;

    fn add(self, other: &Value) -> ValueVec {
        self.add_impl(&ValueVec::from(vec![other.data(); self.len()]))
    }
}

impl<'a> Add<Value> for &'a ValueVec {
    type Output = ValueVec;

    fn add(self, other: Value) -> ValueVec {
        self.add_impl(&ValueVec::from(vec![other.data(); self.len()]))
    }
}

impl<'a> Add<&Value> for &'a ValueVec {
    type Output = ValueVec;

    fn add(self, other: &Value) -> ValueVec {
        self.add_impl(&ValueVec::from(vec![other.data(); self.len()]))
    }
}

impl<'a, 'b> Add<&'a ValueVec> for &'b Value {
    type Output = ValueVec;

    fn add(self, other: &'a ValueVec) -> ValueVec {
        other.add_impl(&ValueVec::from(vec![self.data(); other.len()]))
    }
}

impl<'a> Add<ValueVec> for &'a Value {
    type Output = ValueVec;

    fn add(self, other: ValueVec) -> ValueVec {
        other.add_impl(&ValueVec::from(vec![self.data(); other.len()]))
    }
}

impl Add<&ValueVec> for Value {
    type Output = ValueVec;

    fn add(self, other: &ValueVec) -> ValueVec {
        other.add_impl(&ValueVec::from(vec![self.data(); other.len()]))
    }
}

impl Sub<&ValueVec> for ValueVec {
    type Output = ValueVec;

    fn sub(self, other: &ValueVec) -> ValueVec {
        self + (-other)
    }
}

impl Sub for ValueVec {
    type Output = ValueVec;

    fn sub(self, other: ValueVec) -> ValueVec {
        self + (-other)
    }
}

impl<'a, 'b> Sub<&'a ValueVec> for &'b ValueVec {
    type Output = ValueVec;

    fn sub(self, other: &'a ValueVec) -> ValueVec {
        self + (-other)
    }
}

impl<'a> Sub<ValueVec> for &'a ValueVec {
    type Output = ValueVec;

    fn sub(self, other: ValueVec) -> ValueVec {
        self + (-other)
    }
}

impl Sub<&Value> for ValueVec {
    type Output = ValueVec;

    fn sub(self, other: &Value) -> ValueVec {
        self + (-other)
    }
}

impl Sub<Value> for ValueVec {
    type Output = ValueVec;

    fn sub(self, other: Value) -> ValueVec {
        self + (-other)
    }
}

impl Sub<ValueVec> for Value {
    type Output = ValueVec;

    fn sub(self, other: ValueVec) -> ValueVec {
        self + (-other)
    }
}

impl<'a, 'b> Sub<&'a ValueVec> for &'b Value {
    type Output = ValueVec;

    fn sub(self, other: &'a ValueVec) -> ValueVec {
        self + (-other)
    }
}

impl<'a> Sub<ValueVec> for &'a Value {
    type Output = ValueVec;

    fn sub(self, other: ValueVec) -> ValueVec {
        self + (-other)
    }
}

impl<'a> Sub<Value> for &'a ValueVec {
    type Output = ValueVec;

    fn sub(self, other: Value) -> ValueVec {
        self + (-other)
    }
}

impl<'a> Sub<&Value> for &'a ValueVec {
    type Output = ValueVec;

    fn sub(self, other: &Value) -> ValueVec {
        self + (-other)
    }
}

impl Sub<&ValueVec> for Value {
    type Output = ValueVec;

    fn sub(self, other: &ValueVec) -> ValueVec {
        self + (-other)
    }
}

impl Mul<&ValueVec> for ValueVec {
    type Output = ValueVec;

    fn mul(self, other: &ValueVec) -> ValueVec {
        self.mul_impl(other)
    }
}

impl Mul for ValueVec {
    type Output = ValueVec;

    fn mul(self, other: ValueVec) -> ValueVec {
        self.mul_impl(&other)
    }
}

impl<'a, 'b> Mul<&'a ValueVec> for &'b ValueVec {
    type Output = ValueVec;

    fn mul(self, other: &'a ValueVec) -> ValueVec {
        self.mul_impl(other)
    }
}

impl<'a> Mul<ValueVec> for &'a ValueVec {
    type Output = ValueVec;

    fn mul(self, other: ValueVec) -> ValueVec {
        self.mul_impl(&other)
    }
}

impl Mul<&Value> for ValueVec {
    type Output = ValueVec;

    fn mul(self, other: &Value) -> ValueVec {
        self.mul_impl(&ValueVec::from(vec![other.data(); self.len()]))
    }
}

impl Mul<Value> for ValueVec {
    type Output = ValueVec;

    fn mul(self, other: Value) -> ValueVec {
        self.mul_impl(&ValueVec::from(vec![other.data(); self.len()]))
    }
}

impl Mul<ValueVec> for Value {
    type Output = ValueVec;

    fn mul(self, other: ValueVec) -> ValueVec {
        other.mul_impl(&ValueVec::from(vec![self.data(); other.len()]))
    }
}

impl Mul<&ValueVec> for Value {
    type Output = ValueVec;

    fn mul(self, other: &ValueVec) -> ValueVec {
        other.mul_impl(&ValueVec::from(vec![self.data(); other.len()]))
    }
}

impl<'a, 'b> Mul<&'a ValueVec> for &'b Value {
    type Output = ValueVec;

    fn mul(self, other: &'a ValueVec) -> ValueVec {
        other.mul_impl(&ValueVec::from(vec![self.data(); other.len()]))
    }
}

impl<'a> Mul<ValueVec> for &'a Value {
    type Output = ValueVec;

    fn mul(self, other: ValueVec) -> ValueVec {
        other.mul_impl(&ValueVec::from(vec![self.data(); other.len()]))
    }
}

impl<'a> Mul<Value> for &'a ValueVec {
    type Output = ValueVec;

    fn mul(self, other: Value) -> ValueVec {
        self.mul_impl(&ValueVec::from(vec![other.data(); self.len()]))
    }
}

impl<'a> Mul<&Value> for &'a ValueVec {
    type Output = ValueVec;

    fn mul(self, other: &Value) -> ValueVec {
        self.mul_impl(&ValueVec::from(vec![other.data(); self.len()]))
    }
}

fn assert_non_zero(other: &ValueVec) {
    for val in other.values.iter() {
        if val.data() == 0.0 {
            panic!("Division by zero");
        }
    }
}

impl Div for ValueVec {
    type Output = ValueVec;

    fn div(self, other: ValueVec) -> ValueVec {
        assert_non_zero(&other);
        self * other.pow(&ValueVec::from(vec![-1.0; other.len()]))
    }
}

impl Div<&ValueVec> for ValueVec {
    type Output = ValueVec;

    fn div(self, other: &ValueVec) -> ValueVec {
        assert_non_zero(other);
        self * other.pow(&ValueVec::from(vec![-1.0; other.len()]))
    }
}

impl<'a, 'b> Div<&'a ValueVec> for &'b ValueVec {
    type Output = ValueVec;

    fn div(self, other: &'a ValueVec) -> ValueVec {
        assert_non_zero(other);
        self * other.pow(&ValueVec::from(vec![-1.0; other.len()]))
    }
}

impl<'a> Div<ValueVec> for &'a ValueVec {
    type Output = ValueVec;

    fn div(self, other: ValueVec) -> ValueVec {
        assert_non_zero(&other);
        self * other.pow(&ValueVec::from(vec![-1.0; other.len()]))
    }
}

impl Div<Value> for ValueVec {
    type Output = ValueVec;

    fn div(self, other: Value) -> ValueVec {
        if other.data() == 0.0 {
            panic!("Division by zero");
        }
        let len = self.len();
        self / ValueVec::from(vec![other.data(); len])
    }
}

impl Div<ValueVec> for Value {
    type Output = ValueVec;

    fn div(self, other: ValueVec) -> ValueVec {
        assert_non_zero(&other);
        let len = other.len();
        ValueVec::from(vec![self.data(); len]) / other
    }
}

impl Div<&Value> for ValueVec {
    type Output = ValueVec;

    fn div(self, other: &Value) -> ValueVec {
        if other.data() == 0.0 {
            panic!("Division by zero");
        }
        let len = self.len();
        self / ValueVec::from(vec![other.data(); len])
    }
}

impl<'a, 'b> Div<&'a ValueVec> for &'b Value {
    type Output = ValueVec;

    fn div(self, other: &'a ValueVec) -> ValueVec {
        assert_non_zero(other);
        let len = other.len();
        ValueVec::from(vec![self.data(); len]) / other
    }
}

impl<'a, 'b> Div<&'a Value> for &'b ValueVec {
    type Output = ValueVec;

    fn div(self, other: &'a Value) -> ValueVec {
        if other.data() == 0.0 {
            panic!("Division by zero");
        }
        let len = self.len();
        self / ValueVec::from(vec![other.data(); len])
    }
}

impl<'a> Div<ValueVec> for &'a Value {
    type Output = ValueVec;

    fn div(self, other: ValueVec) -> ValueVec {
        assert_non_zero(&other);
        let len = other.len();
        ValueVec::from(vec![self.data(); len]) / other
    }
}

impl<'a> Div<Value> for &'a ValueVec {
    type Output = ValueVec;

    fn div(self, other: Value) -> ValueVec {
        if other.data() == 0.0 {
            panic!("Division by zero");
        }
        let len = self.len();
        self / ValueVec::from(vec![other.data(); len])
    }
}

impl Div<&ValueVec> for Value {
    type Output = ValueVec;

    fn div(self, other: &ValueVec) -> ValueVec {
        assert_non_zero(&other);
        let len = other.len();
        ValueVec::from(vec![self.data(); len]) / other
    }
}

impl Neg for ValueVec {
    type Output = ValueVec;

    fn neg(self) -> ValueVec {
        let len = self.len();
        self * ValueVec::from(vec![-1.0; len])
    }
}

impl<'a> Neg for &'a ValueVec {
    type Output = ValueVec;

    fn neg(self) -> ValueVec {
        let len = self.len();
        self * ValueVec::from(vec![-1.0; len])
    }
}

impl IntoIterator for ValueVec {
    type Item = Value;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.into_iter()
    }
}

impl<'a> IntoIterator for &'a ValueVec {
    type Item = &'a Value;
    type IntoIter = std::slice::Iter<'a, Value>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.iter()
    }
}

pub struct ValueVecIter<'a> {
    values: &'a ValueVec,
    index: usize,
}

impl<'a> Iterator for ValueVecIter<'a> {
    type Item = &'a Value;

    fn next(&mut self) -> Option<Self::Item> {
        let val = self.values.values.get(self.index);
        self.index += 1;
        val
    }
}

impl FromIterator<Value> for ValueVec {
    fn from_iter<I: IntoIterator<Item = Value>>(iter: I) -> Self {
        let mut v = ValueVec::new();
        for i in iter {
            v.push(i);
        }
        v
    }
}

impl FromIterator<ValueVec> for ValueVec {
    fn from_iter<I: IntoIterator<Item = ValueVec>>(iter: I) -> Self {
        let mut v = ValueVec::new();
        for i in iter {
            for j in i {
                v.push(j);
            }
        }
        v
    }
}
