use std::{
    cell::{Ref, RefCell, RefMut},
    collections::HashSet,
    f64::consts::LN_10,
    fmt,
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
    rc::Rc,
    vec,
};

thread_local! {
    static VALUE_ID: RefCell<usize> = RefCell::new(0);
}

fn gen_id() -> usize {
    VALUE_ID.with(|value_id| {
        let mut value_id = value_id.borrow_mut();
        *value_id += 1;
        *value_id
    })
}

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
    value_id: usize,
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
        visited: &mut HashSet<usize>,
        path: &mut Vec<Self>,
    ) -> () {
        if visited.contains(&node.borrow().value_id) {
            return;
        }
        visited.insert(node.borrow().value_id);
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
        let mut visited = HashSet::<usize>::new();
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
            value_id: gen_id(),
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
            value_id: gen_id(),
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
            value_id: gen_id(),
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
            value_id: gen_id(),
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

impl AddAssign<&Value> for Value {
    fn add_assign(&mut self, other: &Value) {
        *self = self.clone() + other;
    }
}

impl AddAssign for Value {
    fn add_assign(&mut self, other: Value) {
        *self = self.clone() + other;
    }
}

impl SubAssign<&Value> for Value {
    fn sub_assign(&mut self, other: &Value) {
        *self = self.clone() - other;
    }
}

impl SubAssign for Value {
    fn sub_assign(&mut self, other: Value) {
        *self = self.clone() - other;
    }
}

impl MulAssign<&Value> for Value {
    fn mul_assign(&mut self, other: &Value) {
        *self = self.clone() * other;
    }
}

impl MulAssign for Value {
    fn mul_assign(&mut self, other: Value) {
        *self = self.clone() * other;
    }
}

impl DivAssign<&Value> for Value {
    fn div_assign(&mut self, other: &Value) {
        *self = self.clone() / other;
    }
}

impl DivAssign for Value {
    fn div_assign(&mut self, other: Value) {
        *self = self.clone() / other;
    }
}

impl AddAssign<&Value> for Value1d {
    fn add_assign(&mut self, other: &Value) {
        *self = self.clone() + other;
    }
}

impl AddAssign<&Value> for Value2d {
    fn add_assign(&mut self, other: &Value) {
        *self = self.clone() + other;
    }
}

impl AddAssign<Value> for Value1d {
    fn add_assign(&mut self, other: Value) {
        *self = self.clone() + other;
    }
}

impl AddAssign<Value> for Value2d {
    fn add_assign(&mut self, other: Value) {
        *self = self.clone() + other;
    }
}

impl SubAssign<&Value> for Value1d {
    fn sub_assign(&mut self, other: &Value) {
        *self = self.clone() - other;
    }
}

impl SubAssign<&Value> for Value2d {
    fn sub_assign(&mut self, other: &Value) {
        *self = self.clone() - other;
    }
}

impl SubAssign<Value> for Value1d {
    fn sub_assign(&mut self, other: Value) {
        *self = self.clone() - other;
    }
}

impl SubAssign<Value> for Value2d {
    fn sub_assign(&mut self, other: Value) {
        *self = self.clone() - other;
    }
}

impl MulAssign<&Value> for Value1d {
    fn mul_assign(&mut self, other: &Value) {
        *self = self.clone() * other;
    }
}

impl MulAssign<&Value> for Value2d {
    fn mul_assign(&mut self, other: &Value) {
        *self = self.clone() * other;
    }
}

impl MulAssign<Value> for Value1d {
    fn mul_assign(&mut self, other: Value) {
        *self = self.clone() * other;
    }
}

impl MulAssign<Value> for Value2d {
    fn mul_assign(&mut self, other: Value) {
        *self = self.clone() * other;
    }
}

impl DivAssign<&Value> for Value1d {
    fn div_assign(&mut self, other: &Value) {
        *self = self.clone() / other;
    }
}

impl DivAssign<&Value> for Value2d {
    fn div_assign(&mut self, other: &Value) {
        *self = self.clone() / other;
    }
}

impl DivAssign<Value> for Value1d {
    fn div_assign(&mut self, other: Value) {
        *self = self.clone() / other;
    }
}

impl DivAssign<Value> for Value2d {
    fn div_assign(&mut self, other: Value) {
        *self = self.clone() / other;
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
pub struct Value1d {
    values: Vec<Value>,
}

impl Value1d {
    pub fn new() -> Self {
        Self {
            values: Vec::<Value>::new(),
        }
    }

    pub fn zeros(size: usize) -> Self {
        Value1d::from(vec![0.0; size])
    }

    pub fn ones(size: usize) -> Self {
        Value1d::from(vec![1.0; size])
    }

    pub fn rand(size: usize) -> Self {
        let rands = (0..size)
            .map(|_| crate::gen_range(-1.0, 1.0))
            .collect::<Vec<f64>>();
        Value1d::from(rands)
    }

    pub fn rand_range(size: usize, (low, high): (f64, f64)) -> Self {
        let rands = (0..size)
            .map(|_| crate::gen_range(low, high))
            .collect::<Vec<f64>>();
        Value1d::from(rands)
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

    pub fn iter(&self) -> Value1dIter {
        Value1dIter {
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
        assert!(!self.is_empty(), "empty vector");
        self.sum() / Value::from(self.len())
    }

    pub fn mse(&self, other: &Self) -> Value {
        assert!(!self.is_empty(), "empty vector");
        assert_eq!(self.len(), other.len());
        let mut res = Value::from(0.0);
        for (a, b) in self.values.iter().zip(other.values.iter()) {
            res = res + (a - b).pow(&Value::from(2.0));
        }
        res / Value::from(self.len())
    }

    pub fn argmax(&self) -> usize {
        assert!(!self.is_empty(), "empty vector");
        let mut max = Option::<&Value>::None;
        let mut maxidx = 0;
        for (idx, val) in self.iter().enumerate() {
            match max {
                Some(v) if val.data() > v.data() => (max, maxidx) = (Some(val), idx),
                Some(_) => (),
                None => (max, maxidx) = (Some(val), idx),
            }
        }
        maxidx
    }

    pub fn max(&self) -> Value {
        self.values[self.argmax()].clone()
    }

    pub fn argmin(&self) -> usize {
        assert!(!self.is_empty(), "empty vector");
        let mut min = Option::<&Value>::None;
        let mut minidx = 0;
        for (idx, val) in self.iter().enumerate() {
            match min {
                Some(v) if val.data() < v.data() => (min, minidx) = (Some(val), idx),
                Some(_) => (),
                None => (min, minidx) = (Some(val), idx),
            }
        }
        minidx
    }

    pub fn min(&self) -> Value {
        self.values[self.argmin()].clone()
    }

    pub fn softmax(&self) -> Self {
        assert!(!self.is_empty(), "empty vector");
        let exps = self.exp();
        let ref sum = exps.values.iter().sum::<Value>();
        let mut res = Self::new();
        for exp in exps {
            res.push(exp / sum);
        }
        res
    }

    pub fn pow(&self, other: &Self) -> Self {
        assert_eq!(self.len(), other.len(), "different vector shapes");
        let mut res = Self::new();
        for (a, b) in self.values.iter().zip(other.values.iter()) {
            res.push(a.pow(b));
        }
        res
    }

    pub fn dot(&self, other: &Self) -> Value {
        assert_eq!(self.len(), other.len(), "different vector shapes");
        self.values
            .iter()
            .zip(other.values.iter())
            .fold(Value::from(0.0), |acc, (a, b)| acc + (a * b))
    }

    pub fn add_impl(&self, other: &Self) -> Self {
        assert_eq!(self.len(), other.len(), "different vector shapes");
        let mut res = Self::new();
        for (a, b) in self.values.iter().zip(other.values.iter()) {
            res.push(a.add_impl(b));
        }
        res
    }

    pub fn mul_impl(&self, other: &Self) -> Self {
        assert_eq!(self.len(), other.len(), "different vector shapes");
        let mut res = Self::new();
        for (a, b) in self.values.iter().zip(other.values.iter()) {
            res.push(a.mul_impl(b));
        }
        res
    }
}

impl From<Vec<f64>> for Value1d {
    fn from(other: Vec<f64>) -> Value1d {
        let mut v = Value1d::new();
        other
            .into_iter()
            .for_each(|val| v.values.push(Value::from(val)));
        v
    }
}

impl Index<usize> for Value1d {
    type Output = Value;

    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index]
    }
}

impl IndexMut<usize> for Value1d {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.values[index]
    }
}

impl Add<&Value1d> for Value1d {
    type Output = Value1d;

    fn add(self, other: &Value1d) -> Value1d {
        self.add_impl(other)
    }
}

impl Add for Value1d {
    type Output = Value1d;

    fn add(self, other: Value1d) -> Value1d {
        self.add_impl(&other)
    }
}

impl<'a, 'b> Add<&'a Value1d> for &'b Value1d {
    type Output = Value1d;

    fn add(self, other: &'a Value1d) -> Value1d {
        self.add_impl(other)
    }
}

impl<'a> Add<Value1d> for &'a Value1d {
    type Output = Value1d;

    fn add(self, other: Value1d) -> Value1d {
        self.add_impl(&other)
    }
}

impl Add<Value> for Value1d {
    type Output = Value1d;

    fn add(self, other: Value) -> Value1d {
        self.add_impl(&Value1d::from(vec![other.data(); self.len()]))
    }
}

impl Add<Value1d> for Value {
    type Output = Value1d;

    fn add(self, other: Value1d) -> Value1d {
        other.add_impl(&Value1d::from(vec![self.data(); other.len()]))
    }
}

impl Add<&Value> for Value1d {
    type Output = Value1d;

    fn add(self, other: &Value) -> Value1d {
        self.add_impl(&Value1d::from(vec![other.data(); self.len()]))
    }
}

impl<'a> Add<Value> for &'a Value1d {
    type Output = Value1d;

    fn add(self, other: Value) -> Value1d {
        self.add_impl(&Value1d::from(vec![other.data(); self.len()]))
    }
}

impl<'a> Add<&Value> for &'a Value1d {
    type Output = Value1d;

    fn add(self, other: &Value) -> Value1d {
        self.add_impl(&Value1d::from(vec![other.data(); self.len()]))
    }
}

impl<'a, 'b> Add<&'a Value1d> for &'b Value {
    type Output = Value1d;

    fn add(self, other: &'a Value1d) -> Value1d {
        other.add_impl(&Value1d::from(vec![self.data(); other.len()]))
    }
}

impl<'a> Add<Value1d> for &'a Value {
    type Output = Value1d;

    fn add(self, other: Value1d) -> Value1d {
        other.add_impl(&Value1d::from(vec![self.data(); other.len()]))
    }
}

impl Add<&Value1d> for Value {
    type Output = Value1d;

    fn add(self, other: &Value1d) -> Value1d {
        other.add_impl(&Value1d::from(vec![self.data(); other.len()]))
    }
}

impl Sub<&Value1d> for Value1d {
    type Output = Value1d;

    fn sub(self, other: &Value1d) -> Value1d {
        self + (-other)
    }
}

impl Sub for Value1d {
    type Output = Value1d;

    fn sub(self, other: Value1d) -> Value1d {
        self + (-other)
    }
}

impl<'a, 'b> Sub<&'a Value1d> for &'b Value1d {
    type Output = Value1d;

    fn sub(self, other: &'a Value1d) -> Value1d {
        self + (-other)
    }
}

impl<'a> Sub<Value1d> for &'a Value1d {
    type Output = Value1d;

    fn sub(self, other: Value1d) -> Value1d {
        self + (-other)
    }
}

impl Sub<&Value> for Value1d {
    type Output = Value1d;

    fn sub(self, other: &Value) -> Value1d {
        self + (-other)
    }
}

impl Sub<Value> for Value1d {
    type Output = Value1d;

    fn sub(self, other: Value) -> Value1d {
        self + (-other)
    }
}

impl Sub<Value1d> for Value {
    type Output = Value1d;

    fn sub(self, other: Value1d) -> Value1d {
        self + (-other)
    }
}

impl<'a, 'b> Sub<&'a Value1d> for &'b Value {
    type Output = Value1d;

    fn sub(self, other: &'a Value1d) -> Value1d {
        self + (-other)
    }
}

impl<'a> Sub<Value1d> for &'a Value {
    type Output = Value1d;

    fn sub(self, other: Value1d) -> Value1d {
        self + (-other)
    }
}

impl<'a> Sub<Value> for &'a Value1d {
    type Output = Value1d;

    fn sub(self, other: Value) -> Value1d {
        self + (-other)
    }
}

impl<'a> Sub<&Value> for &'a Value1d {
    type Output = Value1d;

    fn sub(self, other: &Value) -> Value1d {
        self + (-other)
    }
}

impl Sub<&Value1d> for Value {
    type Output = Value1d;

    fn sub(self, other: &Value1d) -> Value1d {
        self + (-other)
    }
}

impl Mul<&Value1d> for Value1d {
    type Output = Value1d;

    fn mul(self, other: &Value1d) -> Value1d {
        self.mul_impl(other)
    }
}

impl Mul for Value1d {
    type Output = Value1d;

    fn mul(self, other: Value1d) -> Value1d {
        self.mul_impl(&other)
    }
}

impl<'a, 'b> Mul<&'a Value1d> for &'b Value1d {
    type Output = Value1d;

    fn mul(self, other: &'a Value1d) -> Value1d {
        self.mul_impl(other)
    }
}

impl<'a> Mul<Value1d> for &'a Value1d {
    type Output = Value1d;

    fn mul(self, other: Value1d) -> Value1d {
        self.mul_impl(&other)
    }
}

impl Mul<&Value> for Value1d {
    type Output = Value1d;

    fn mul(self, other: &Value) -> Value1d {
        self.mul_impl(&Value1d::from(vec![other.data(); self.len()]))
    }
}

impl Mul<Value> for Value1d {
    type Output = Value1d;

    fn mul(self, other: Value) -> Value1d {
        self.mul_impl(&Value1d::from(vec![other.data(); self.len()]))
    }
}

impl Mul<Value1d> for Value {
    type Output = Value1d;

    fn mul(self, other: Value1d) -> Value1d {
        other.mul_impl(&Value1d::from(vec![self.data(); other.len()]))
    }
}

impl Mul<&Value1d> for Value {
    type Output = Value1d;

    fn mul(self, other: &Value1d) -> Value1d {
        other.mul_impl(&Value1d::from(vec![self.data(); other.len()]))
    }
}

impl<'a, 'b> Mul<&'a Value1d> for &'b Value {
    type Output = Value1d;

    fn mul(self, other: &'a Value1d) -> Value1d {
        other.mul_impl(&Value1d::from(vec![self.data(); other.len()]))
    }
}

impl<'a> Mul<Value1d> for &'a Value {
    type Output = Value1d;

    fn mul(self, other: Value1d) -> Value1d {
        other.mul_impl(&Value1d::from(vec![self.data(); other.len()]))
    }
}

impl<'a> Mul<Value> for &'a Value1d {
    type Output = Value1d;

    fn mul(self, other: Value) -> Value1d {
        self.mul_impl(&Value1d::from(vec![other.data(); self.len()]))
    }
}

impl<'a> Mul<&Value> for &'a Value1d {
    type Output = Value1d;

    fn mul(self, other: &Value) -> Value1d {
        self.mul_impl(&Value1d::from(vec![other.data(); self.len()]))
    }
}

fn assert_div_non_zero(other: &Value1d) {
    for val in other.values.iter() {
        assert!(val.data() != 0.0, "Division by zero");
    }
}

impl Div for Value1d {
    type Output = Value1d;

    fn div(self, other: Value1d) -> Value1d {
        assert_div_non_zero(&other);
        self * other.pow(&Value1d::from(vec![-1.0; other.len()]))
    }
}

impl Div<&Value1d> for Value1d {
    type Output = Value1d;

    fn div(self, other: &Value1d) -> Value1d {
        assert_div_non_zero(other);
        self * other.pow(&Value1d::from(vec![-1.0; other.len()]))
    }
}

impl<'a, 'b> Div<&'a Value1d> for &'b Value1d {
    type Output = Value1d;

    fn div(self, other: &'a Value1d) -> Value1d {
        assert_div_non_zero(other);
        self * other.pow(&Value1d::from(vec![-1.0; other.len()]))
    }
}

impl<'a> Div<Value1d> for &'a Value1d {
    type Output = Value1d;

    fn div(self, other: Value1d) -> Value1d {
        assert_div_non_zero(&other);
        self * other.pow(&Value1d::from(vec![-1.0; other.len()]))
    }
}

impl Div<Value> for Value1d {
    type Output = Value1d;

    fn div(self, other: Value) -> Value1d {
        assert!(other.data() != 0.0, "Division by zero");
        let len = self.len();
        self / Value1d::from(vec![other.data(); len])
    }
}

impl Div<Value1d> for Value {
    type Output = Value1d;

    fn div(self, other: Value1d) -> Value1d {
        assert_div_non_zero(&other);
        let len = other.len();
        Value1d::from(vec![self.data(); len]) / other
    }
}

impl Div<&Value> for Value1d {
    type Output = Value1d;

    fn div(self, other: &Value) -> Value1d {
        assert!(other.data() != 0.0, "Division by zero");
        let len = self.len();
        self / Value1d::from(vec![other.data(); len])
    }
}

impl<'a, 'b> Div<&'a Value1d> for &'b Value {
    type Output = Value1d;

    fn div(self, other: &'a Value1d) -> Value1d {
        assert_div_non_zero(other);
        let len = other.len();
        Value1d::from(vec![self.data(); len]) / other
    }
}

impl<'a, 'b> Div<&'a Value> for &'b Value1d {
    type Output = Value1d;

    fn div(self, other: &'a Value) -> Value1d {
        assert!(other.data() != 0.0, "Division by zero");
        let len = self.len();
        self / Value1d::from(vec![other.data(); len])
    }
}

impl<'a> Div<Value1d> for &'a Value {
    type Output = Value1d;

    fn div(self, other: Value1d) -> Value1d {
        assert_div_non_zero(&other);
        let len = other.len();
        Value1d::from(vec![self.data(); len]) / other
    }
}

impl<'a> Div<Value> for &'a Value1d {
    type Output = Value1d;

    fn div(self, other: Value) -> Value1d {
        assert!(other.data() != 0.0, "Division by zero");
        let len = self.len();
        self / Value1d::from(vec![other.data(); len])
    }
}

impl Div<&Value1d> for Value {
    type Output = Value1d;

    fn div(self, other: &Value1d) -> Value1d {
        assert_div_non_zero(&other);
        let len = other.len();
        Value1d::from(vec![self.data(); len]) / other
    }
}

impl Neg for Value1d {
    type Output = Value1d;

    fn neg(self) -> Value1d {
        let len = self.len();
        self * Value1d::from(vec![-1.0; len])
    }
}

impl<'a> Neg for &'a Value1d {
    type Output = Value1d;

    fn neg(self) -> Value1d {
        let len = self.len();
        self * Value1d::from(vec![-1.0; len])
    }
}

impl AddAssign<&Value1d> for Value1d {
    fn add_assign(&mut self, other: &Value1d) {
        *self = self.clone() + other;
    }
}

impl AddAssign for Value1d {
    fn add_assign(&mut self, other: Value1d) {
        *self = self.clone() + other;
    }
}

impl SubAssign<&Value1d> for Value1d {
    fn sub_assign(&mut self, other: &Value1d) {
        *self = self.clone() - other;
    }
}

impl SubAssign for Value1d {
    fn sub_assign(&mut self, other: Value1d) {
        *self = self.clone() - other;
    }
}

impl MulAssign<&Value1d> for Value1d {
    fn mul_assign(&mut self, other: &Value1d) {
        *self = self.clone() * other;
    }
}

impl MulAssign for Value1d {
    fn mul_assign(&mut self, other: Value1d) {
        *self = self.clone() * other;
    }
}

impl DivAssign<&Value1d> for Value1d {
    fn div_assign(&mut self, other: &Value1d) {
        *self = self.clone() / other;
    }
}

impl DivAssign for Value1d {
    fn div_assign(&mut self, other: Value1d) {
        *self = self.clone() / other;
    }
}

impl IntoIterator for Value1d {
    type Item = Value;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.into_iter()
    }
}

impl<'a> IntoIterator for &'a Value1d {
    type Item = &'a Value;
    type IntoIter = std::slice::Iter<'a, Value>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.iter()
    }
}

pub struct Value1dIter<'a> {
    values: &'a Value1d,
    index: usize,
}

impl<'a> Iterator for Value1dIter<'a> {
    type Item = &'a Value;

    fn next(&mut self) -> Option<Self::Item> {
        let val = self.values.values.get(self.index);
        self.index += 1;
        val
    }
}

impl FromIterator<Value> for Value1d {
    fn from_iter<I: IntoIterator<Item = Value>>(iter: I) -> Self {
        let mut v = Value1d::new();
        for i in iter {
            v.push(i);
        }
        v
    }
}

impl FromIterator<Value1d> for Value1d {
    fn from_iter<I: IntoIterator<Item = Value1d>>(iter: I) -> Self {
        let mut v = Value1d::new();
        for i in iter {
            for j in i {
                v.push(j);
            }
        }
        v
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd)]
pub struct Value2d {
    values: Vec<Value1d>,
}

impl Value2d {
    pub fn new() -> Self {
        Self {
            values: Vec::<Value1d>::new(),
        }
    }

    pub fn rand((rows, cols): (usize, usize)) -> Self {
        let mut res = Self::zeros((rows, cols));
        for i in 0..rows {
            for j in 0..cols {
                res[(i, j)] = Value::from(crate::gen_range(-1.0, 1.0));
            }
        }
        res
    }

    pub fn data(&self) -> Vec<Vec<f64>> {
        self.values.iter().map(|v| v.data()).collect()
    }

    pub fn grad(&self) -> Vec<Vec<f64>> {
        self.values.iter().map(|v| v.grad()).collect()
    }

    pub fn update(&mut self, rate: f64) {
        self.values.iter_mut().for_each(|v| v.update(rate));
    }

    pub fn zero_grad(&mut self) {
        self.values.iter_mut().for_each(|v| v.zero_grad());
    }

    pub fn backward(&self) {
        self.values.iter().for_each(|v| v.backward());
    }

    pub fn len(&self) -> usize {
        let (rows, cols) = self.shape();
        return rows * cols;
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn iter(&self) -> Value2dIter {
        Value2dIter {
            values: self,
            index: 0,
            shape: self.shape(),
        }
    }

    pub fn zeros((rows, cols): (usize, usize)) -> Self {
        let mut res = Self::new();
        for _ in 0..rows {
            res.values.push(Value1d::zeros(cols));
        }
        res
    }

    pub fn ones((rows, cols): (usize, usize)) -> Self {
        let mut res = Self::new();
        for _ in 0..rows {
            res.values.push(Value1d::ones(cols));
        }
        res
    }

    pub fn from_value(value: Value, (rows, cols): (usize, usize)) -> Self {
        let mut res = Self::new();
        for _ in 0..rows {
            let mut row = Value1d::new();
            for _ in 0..cols {
                row.push(value.clone());
            }
            res.values.push(row);
        }
        res
    }

    pub fn shape(&self) -> (usize, usize) {
        if self.values.len() == 0 {
            return (0, 0);
        }
        (self.values.len(), self.values[0].len())
    }

    pub fn relu(&self) -> Self {
        let mut res = Self::new();
        for val in self.values.iter() {
            res.values.push(val.relu());
        }
        res
    }

    pub fn tanh(&self) -> Self {
        let mut res = Self::new();
        for val in self.values.iter() {
            res.values.push(val.tanh());
        }
        res
    }

    pub fn exp(&self) -> Self {
        let mut res = Self::new();
        for val in self.values.iter() {
            res.values.push(val.exp());
        }
        res
    }

    pub fn log10(&self) -> Self {
        let mut res = Self::new();
        for val in self.values.iter() {
            res.values.push(val.log10());
        }
        res
    }

    pub fn log(&self) -> Self {
        let mut res = Self::new();
        for val in self.values.iter() {
            res.values.push(val.log());
        }
        res
    }

    pub fn sigmoid(&self) -> Self {
        let mut res = Self::new();
        for val in self.values.iter() {
            res.values.push(val.sigmoid());
        }
        res
    }

    pub fn sum(&self) -> Value {
        let mut res = Value::from(0.0);
        for row in self.values.iter() {
            res = res + row.sum();
        }
        res
    }

    pub fn sum_axis_1(&self) -> Value1d {
        let mut res = Value1d::new();
        for row in self.values.iter() {
            res.values.push(row.sum());
        }
        res
    }

    pub fn sum_axis_0(&self) -> Value1d {
        let mut res = Value1d::new();
        let t = self.transpose();
        for row in t.values.iter() {
            res.values.push(row.sum());
        }
        res
    }

    pub fn softmax(&self) -> Self {
        assert!(!self.is_empty(), "empty Value2D");
        let exps = self.exp();
        let ref sum = exps.iter().sum::<Value>();
        let mut res = Self::zeros(self.shape());
        for i in 0..self.shape().0 {
            for j in 0..self.shape().1 {
                res[(i, j)] = exps[(i, j)].clone() / sum;
            }
        }
        res
    }

    pub fn softmax_axis_1(&self) -> Self {
        let mut res = Self::new();
        for val in self.values.iter() {
            res.values.push(val.softmax());
        }
        res
    }

    pub fn softmax_axis_0(&self) -> Self {
        let mut res = Self::new();
        let t = self.transpose();
        for val in t.values.iter() {
            res.values.push(val.softmax());
        }
        res.transpose()
    }

    pub fn pow(&self, other: &Self) -> Self {
        assert_eq!(self.shape(), other.shape(), "different Value2D shapes");
        let mut res = Self::new();
        for (a, b) in self.values.iter().zip(other.values.iter()) {
            res.values.push(a.pow(b));
        }
        res
    }

    pub fn argmax(&self) -> (usize, usize) {
        assert!(!self.values.is_empty(), "empty Value2D");
        let mut max = Option::<&Value>::None;
        let mut maxidx = 0;
        for (idx, val) in self.iter().enumerate() {
            match max {
                Some(v) if val.data() > v.data() => (max, maxidx) = (Some(val), idx),
                Some(_) => (),
                None => (max, maxidx) = (Some(val), idx),
            }
        }
        (maxidx / self.shape().1, maxidx % self.shape().1)
    }

    pub fn max(&self) -> Value {
        self.values[self.argmax().0][self.argmax().1].clone()
    }

    pub fn argmax_axis_1(&self) -> Vec<(usize, usize)> {
        assert!(!self.values.is_empty(), "empty Value2D");
        let mut res = Vec::<(usize, usize)>::new();
        for row in self.values.iter() {
            res.push((res.len(), row.argmax()));
        }
        res
    }

    pub fn argmax_axis_0(&self) -> Vec<(usize, usize)> {
        assert!(!self.values.is_empty(), "empty Value2D");
        let mut res = Vec::<(usize, usize)>::new();
        let t = self.transpose();
        for row in t.values.iter() {
            res.push((row.argmax(), res.len()));
        }
        res
    }

    pub fn max_axis_1(&self) -> Value1d {
        assert!(!self.values.is_empty(), "empty Value2D");
        let mut res = Value1d::new();
        for row in self.values.iter() {
            res.push(row.max());
        }
        res
    }

    pub fn max_axis_0(&self) -> Value1d {
        assert!(!self.values.is_empty(), "empty Value2D");
        let mut res = Value1d::new();
        let t = self.transpose();
        for row in t.values.iter() {
            res.push(row.max());
        }
        res
    }

    pub fn argmin(&self) -> (usize, usize) {
        assert!(!self.values.is_empty(), "empty Value2D");
        let mut min = Option::<&Value>::None;
        let mut minidx = 0;
        for (idx, val) in self.iter().enumerate() {
            match min {
                Some(v) if val.data() < v.data() => (min, minidx) = (Some(val), idx),
                Some(_) => (),
                None => (min, minidx) = (Some(val), idx),
            }
        }
        (minidx / self.shape().1, minidx % self.shape().1)
    }

    pub fn min(&self) -> Value {
        self.values[self.argmin().0][self.argmin().1].clone()
    }

    pub fn argmin_axis_1(&self) -> Vec<(usize, usize)> {
        assert!(!self.values.is_empty(), "empty Value2D");
        let mut res = Vec::<(usize, usize)>::new();
        for row in self.values.iter() {
            res.push((res.len(), row.argmin()));
        }
        res
    }

    pub fn argmin_axis_0(&self) -> Vec<(usize, usize)> {
        assert!(!self.values.is_empty(), "empty Value2D");
        let mut res = Vec::<(usize, usize)>::new();
        let t = self.transpose();
        for row in t.values.iter() {
            res.push((row.argmin(), res.len()));
        }
        res
    }

    pub fn min_axis_1(&self) -> Value1d {
        assert!(!self.values.is_empty(), "empty Value2D");
        let mut res = Value1d::new();
        for row in self.values.iter() {
            res.push(row.min());
        }
        res
    }

    pub fn min_axis_0(&self) -> Value1d {
        assert!(!self.values.is_empty(), "empty Value2D");
        let mut res = Value1d::new();
        let t = self.transpose();
        for row in t.values.iter() {
            res.push(row.min());
        }
        res
    }

    pub fn mean(&self) -> Value {
        assert!(!self.values.is_empty(), "empty Value2D");
        self.sum() / Value::from(self.len())
    }

    pub fn mean_axis_1(&self) -> Value1d {
        assert!(!self.values.is_empty(), "empty Value2D");
        self.sum_axis_1() / Value::from(self.shape().1)
    }

    pub fn mean_axis_0(&self) -> Value1d {
        assert!(!self.values.is_empty(), "empty Value2D");
        self.sum_axis_0() / Value::from(self.shape().0)
    }

    pub fn mse(&self, other: &Self) -> Value {
        assert!(!self.is_empty(), "empty Value2D");
        assert_eq!(self.shape(), other.shape(), "different Value2D shapes");
        let mut res = Value::from(0.0);
        for (a, b) in self.iter().zip(other.iter()) {
            res = res + (a - b).pow(&Value::from(2.0));
        }
        res / Value::from(self.len())
    }

    pub fn mse_axis_1(&self, other: &Self) -> Value1d {
        assert!(!self.is_empty(), "empty Value2D");
        assert_eq!(self.shape(), other.shape(), "different Value2D shapes");
        let mut res = Value1d::new();
        for (a, b) in self.values.iter().zip(other.values.iter()) {
            res.push(a.mse(b));
        }
        res
    }

    pub fn mse_axis_0(&self, other: &Self) -> Value1d {
        assert!(!self.is_empty(), "empty Value2D");
        assert_eq!(self.shape(), other.shape(), "different Value2D shapes");
        let mut res = Value1d::new();
        let t = self.transpose();
        let other_t = other.transpose();
        for (a, b) in t.values.iter().zip(other_t.values.iter()) {
            res.push(a.mse(b));
        }
        res
    }

    pub fn transpose(&self) -> Self {
        let shape = self.shape();
        let mut res = Self::from_value(Value::from(0.0), (shape.1, shape.0));
        for i in 0..self.shape().0 {
            for j in 0..self.shape().1 {
                res[(j, i)] = self[(i, j)].clone();
            }
        }
        res
    }

    pub fn matmul(&self, other: &Self) -> Self {
        let mut res = Self::new();
        let other_t = other.transpose();
        for row in self.values.iter() {
            let mut row_res = Value1d::new();
            for col in other_t.values.iter() {
                row_res.push(row.dot(col));
            }
            res.values.push(row_res);
        }
        res
    }

    pub fn add_impl(&self, other: &Self) -> Self {
        assert_eq!(self.shape(), other.shape(), "different Value2D shapes");
        let mut res = Self::new();
        for (a, b) in self.values.iter().zip(other.values.iter()) {
            res.values.push(a.add_impl(b));
        }
        res
    }

    pub fn mul_impl(&self, other: &Self) -> Self {
        assert_eq!(self.shape(), other.shape(), "different Value2D shapes");
        let mut res = Self::new();
        for (a, b) in self.values.iter().zip(other.values.iter()) {
            res.values.push(a.mul_impl(b));
        }
        res
    }
}

impl From<Vec<Value1d>> for Value2d {
    fn from(other: Vec<Value1d>) -> Value2d {
        let mut v = Value2d::new();
        other.into_iter().for_each(|val| v.values.push(val));
        v
    }
}

impl FromIterator<Value1d> for Value2d {
    fn from_iter<I: IntoIterator<Item = Value1d>>(iter: I) -> Self {
        let mut v = Value2d::new();
        for i in iter {
            v.values.push(i);
        }
        v
    }
}

impl Index<(usize, usize)> for Value2d {
    type Output = Value;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.values[index.0][index.1]
    }
}

impl IndexMut<(usize, usize)> for Value2d {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.values[index.0][index.1]
    }
}

impl Add<&Value2d> for Value2d {
    type Output = Value2d;

    fn add(self, other: &Value2d) -> Value2d {
        self.add_impl(other)
    }
}

impl Add for Value2d {
    type Output = Value2d;

    fn add(self, other: Value2d) -> Value2d {
        self.add_impl(&other)
    }
}

impl<'a, 'b> Add<&'a Value2d> for &'b Value2d {
    type Output = Value2d;

    fn add(self, other: &'a Value2d) -> Value2d {
        self.add_impl(other)
    }
}

impl<'a> Add<Value2d> for &'a Value2d {
    type Output = Value2d;

    fn add(self, other: Value2d) -> Value2d {
        self.add_impl(&other)
    }
}

impl Add<Value> for Value2d {
    type Output = Value2d;

    fn add(self, other: Value) -> Value2d {
        self.add_impl(&Value2d::from_value(other, self.shape()))
    }
}

impl Add<Value2d> for Value {
    type Output = Value2d;

    fn add(self, other: Value2d) -> Value2d {
        other.add_impl(&Value2d::from_value(self, other.shape()))
    }
}

impl Add<&Value> for Value2d {
    type Output = Value2d;

    fn add(self, other: &Value) -> Value2d {
        self.add_impl(&Value2d::from_value(other.clone(), self.shape()))
    }
}

impl<'a> Add<Value> for &'a Value2d {
    type Output = Value2d;

    fn add(self, other: Value) -> Value2d {
        self.add_impl(&Value2d::from_value(other, self.shape()))
    }
}

impl<'a> Add<&Value> for &'a Value2d {
    type Output = Value2d;

    fn add(self, other: &Value) -> Value2d {
        self.add_impl(&Value2d::from_value(other.clone(), self.shape()))
    }
}

impl<'a, 'b> Add<&'a Value2d> for &'b Value {
    type Output = Value2d;

    fn add(self, other: &'a Value2d) -> Value2d {
        other.add_impl(&Value2d::from_value(self.clone(), other.shape()))
    }
}

impl<'a> Add<Value2d> for &'a Value {
    type Output = Value2d;

    fn add(self, other: Value2d) -> Value2d {
        other.add_impl(&Value2d::from_value(self.clone(), other.shape()))
    }
}

impl Add<&Value2d> for Value {
    type Output = Value2d;

    fn add(self, other: &Value2d) -> Value2d {
        other.add_impl(&Value2d::from_value(self, other.shape()))
    }
}

impl Sub<&Value2d> for Value2d {
    type Output = Value2d;

    fn sub(self, other: &Value2d) -> Value2d {
        self + (-other)
    }
}

impl Sub for Value2d {
    type Output = Value2d;

    fn sub(self, other: Value2d) -> Value2d {
        self + (-other)
    }
}

impl<'a, 'b> Sub<&'a Value2d> for &'b Value2d {
    type Output = Value2d;

    fn sub(self, other: &'a Value2d) -> Value2d {
        self + (-other)
    }
}

impl<'a> Sub<Value2d> for &'a Value2d {
    type Output = Value2d;

    fn sub(self, other: Value2d) -> Value2d {
        self + (-other)
    }
}

impl Sub<&Value> for Value2d {
    type Output = Value2d;

    fn sub(self, other: &Value) -> Value2d {
        self + (-other)
    }
}

impl Sub<Value> for Value2d {
    type Output = Value2d;

    fn sub(self, other: Value) -> Value2d {
        self + (-other)
    }
}

impl Sub<Value2d> for Value {
    type Output = Value2d;

    fn sub(self, other: Value2d) -> Value2d {
        self + (-other)
    }
}

impl<'a, 'b> Sub<&'a Value2d> for &'b Value {
    type Output = Value2d;

    fn sub(self, other: &'a Value2d) -> Value2d {
        self + (-other)
    }
}

impl<'a> Sub<Value2d> for &'a Value {
    type Output = Value2d;

    fn sub(self, other: Value2d) -> Value2d {
        self + (-other)
    }
}

impl<'a> Sub<Value> for &'a Value2d {
    type Output = Value2d;

    fn sub(self, other: Value) -> Value2d {
        self + (-other)
    }
}

impl<'a> Sub<&Value> for &'a Value2d {
    type Output = Value2d;

    fn sub(self, other: &Value) -> Value2d {
        self + (-other)
    }
}

impl Sub<&Value2d> for Value {
    type Output = Value2d;

    fn sub(self, other: &Value2d) -> Value2d {
        self + (-other)
    }
}

impl Mul<&Value2d> for Value2d {
    type Output = Value2d;

    fn mul(self, other: &Value2d) -> Value2d {
        self.mul_impl(other)
    }
}

impl Mul for Value2d {
    type Output = Value2d;

    fn mul(self, other: Value2d) -> Value2d {
        self.mul_impl(&other)
    }
}

impl<'a, 'b> Mul<&'a Value2d> for &'b Value2d {
    type Output = Value2d;

    fn mul(self, other: &'a Value2d) -> Value2d {
        self.mul_impl(other)
    }
}

impl<'a> Mul<Value2d> for &'a Value2d {
    type Output = Value2d;

    fn mul(self, other: Value2d) -> Value2d {
        self.mul_impl(&other)
    }
}

impl Mul<&Value> for Value2d {
    type Output = Value2d;

    fn mul(self, other: &Value) -> Value2d {
        self.mul_impl(&Value2d::from_value(other.clone(), self.shape()))
    }
}

impl Mul<Value> for Value2d {
    type Output = Value2d;

    fn mul(self, other: Value) -> Value2d {
        self.mul_impl(&Value2d::from_value(other, self.shape()))
    }
}

impl Mul<Value2d> for Value {
    type Output = Value2d;

    fn mul(self, other: Value2d) -> Value2d {
        other.mul_impl(&Value2d::from_value(self, other.shape()))
    }
}

impl Mul<&Value2d> for Value {
    type Output = Value2d;

    fn mul(self, other: &Value2d) -> Value2d {
        other.mul_impl(&Value2d::from_value(self, other.shape()))
    }
}

impl<'a, 'b> Mul<&'a Value2d> for &'b Value {
    type Output = Value2d;

    fn mul(self, other: &'a Value2d) -> Value2d {
        other.mul_impl(&Value2d::from_value(self.clone(), other.shape()))
    }
}

impl<'a> Mul<Value2d> for &'a Value {
    type Output = Value2d;

    fn mul(self, other: Value2d) -> Value2d {
        other.mul_impl(&Value2d::from_value(self.clone(), other.shape()))
    }
}

impl<'a> Mul<Value> for &'a Value2d {
    type Output = Value2d;

    fn mul(self, other: Value) -> Value2d {
        self.mul_impl(&Value2d::from_value(other, self.shape()))
    }
}

impl<'a> Mul<&Value> for &'a Value2d {
    type Output = Value2d;

    fn mul(self, other: &Value) -> Value2d {
        self.mul_impl(&Value2d::from_value(other.clone(), self.shape()))
    }
}

fn assert_div_non_zero_mat(other: &Value2d) {
    for val in other.values.iter() {
        for val in val.values.iter() {
            assert!(val.data() != 0.0, "Division by zero");
        }
    }
}

impl Div for Value2d {
    type Output = Value2d;

    fn div(self, other: Value2d) -> Value2d {
        assert_div_non_zero_mat(&other);
        self * other.pow(&Value2d::from_value(Value::from(-1.0), other.shape()))
    }
}

impl Div<&Value2d> for Value2d {
    type Output = Value2d;

    fn div(self, other: &Value2d) -> Value2d {
        assert_div_non_zero_mat(other);
        self * other.pow(&Value2d::from_value(Value::from(-1.0), other.shape()))
    }
}

impl<'a, 'b> Div<&'a Value2d> for &'b Value2d {
    type Output = Value2d;

    fn div(self, other: &'a Value2d) -> Value2d {
        assert_div_non_zero_mat(other);
        self * other.pow(&Value2d::from_value(Value::from(-1.0), other.shape()))
    }
}

impl<'a> Div<Value2d> for &'a Value2d {
    type Output = Value2d;

    fn div(self, other: Value2d) -> Value2d {
        assert_div_non_zero_mat(&other);
        self * other.pow(&Value2d::from_value(Value::from(-1.0), other.shape()))
    }
}

impl Div<Value> for Value2d {
    type Output = Value2d;

    fn div(self, other: Value) -> Value2d {
        assert!(other.data() != 0.0, "Division by zero");
        let size = self.shape();
        self / Value2d::from_value(other, size)
    }
}

impl Div<Value2d> for Value {
    type Output = Value2d;

    fn div(self, other: Value2d) -> Value2d {
        assert_div_non_zero_mat(&other);
        Value2d::from_value(self, other.shape()) / other
    }
}

impl Div<&Value> for Value2d {
    type Output = Value2d;

    fn div(self, other: &Value) -> Value2d {
        assert!(other.data() != 0.0, "Division by zero");
        let size = self.shape();
        self / Value2d::from_value(other.clone(), size)
    }
}

impl<'a, 'b> Div<&'a Value2d> for &'b Value {
    type Output = Value2d;

    fn div(self, other: &'a Value2d) -> Value2d {
        assert_div_non_zero_mat(other);
        let size = other.shape();
        Value2d::from_value(self.clone(), size) / other
    }
}

impl<'a, 'b> Div<&'a Value> for &'b Value2d {
    type Output = Value2d;

    fn div(self, other: &'a Value) -> Value2d {
        assert!(other.data() != 0.0, "Division by zero");
        let size = self.shape();
        self / Value2d::from_value(other.clone(), size)
    }
}

impl<'a> Div<Value2d> for &'a Value {
    type Output = Value2d;

    fn div(self, other: Value2d) -> Value2d {
        assert_div_non_zero_mat(&other);
        let size = other.shape();
        Value2d::from_value(self.clone(), size) / other
    }
}

impl<'a> Div<Value> for &'a Value2d {
    type Output = Value2d;

    fn div(self, other: Value) -> Value2d {
        assert!(other.data() != 0.0, "Division by zero");
        let size = self.shape();
        self / Value2d::from_value(other, size)
    }
}

impl Div<&Value2d> for Value {
    type Output = Value2d;

    fn div(self, other: &Value2d) -> Value2d {
        assert_div_non_zero_mat(&other);
        let size = other.shape();
        Value2d::from_value(self, size) / other
    }
}

impl Neg for Value2d {
    type Output = Value2d;

    fn neg(self) -> Value2d {
        let size = self.shape();
        self * Value2d::from_value(Value::from(-1.0), size)
    }
}

impl<'a> Neg for &'a Value2d {
    type Output = Value2d;

    fn neg(self) -> Value2d {
        let size = self.shape();
        self * Value2d::from_value(Value::from(-1.0), size)
    }
}

impl AddAssign<&Value2d> for Value2d {
    fn add_assign(&mut self, other: &Value2d) {
        *self = self.clone() + other;
    }
}

impl AddAssign for Value2d {
    fn add_assign(&mut self, other: Value2d) {
        *self = self.clone() + other;
    }
}

impl SubAssign<&Value2d> for Value2d {
    fn sub_assign(&mut self, other: &Value2d) {
        *self = self.clone() - other;
    }
}

impl SubAssign for Value2d {
    fn sub_assign(&mut self, other: Value2d) {
        *self = self.clone() - other;
    }
}

impl MulAssign<&Value2d> for Value2d {
    fn mul_assign(&mut self, other: &Value2d) {
        *self = self.clone() * other;
    }
}

impl MulAssign for Value2d {
    fn mul_assign(&mut self, other: Value2d) {
        *self = self.clone() * other;
    }
}

impl DivAssign<&Value2d> for Value2d {
    fn div_assign(&mut self, other: &Value2d) {
        *self = self.clone() / other;
    }
}

impl DivAssign for Value2d {
    fn div_assign(&mut self, other: Value2d) {
        *self = self.clone() / other;
    }
}

pub struct Value2dIter<'a> {
    values: &'a Value2d,
    shape: (usize, usize),
    index: usize,
}

impl<'a> Iterator for Value2dIter<'a> {
    type Item = &'a Value;

    fn next(&mut self) -> Option<Self::Item> {
        let (row, col) = (self.index / self.shape.1, self.index % self.shape.1);
        let val = self.values.values.get(row)?.values.get(col);
        self.index += 1;
        val
    }
}

impl IntoIterator for Value2d {
    type Item = Value;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter().cloned().collect::<Vec<_>>().into_iter()
    }
}

impl<'a> IntoIterator for &'a Value2d {
    type Item = &'a Value;
    type IntoIter = Value2dIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
