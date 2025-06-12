use microgradr::{gnn::MessagePassingTrait, gnn::GCN, gnn::SAGE, *};

#[test]
fn test_message_passing_basic() {
    // Create a simple graph with 3 nodes and 2 edges
    // Edge list: (0->1), (1->2)
    let edge_index = v2d![
        v1d![0.0, 1.0], // source nodes
        v1d![1.0, 2.0]  // target nodes
    ];

    // Node features: 3 nodes, 2 features each
    let node_features = v2d![
        v1d![1.0, 2.0], // node 0
        v1d![3.0, 4.0], // node 1
        v1d![5.0, 6.0]  // node 2
    ];

    let mp = MessagePassing::new();

    // Test message creation for batched edges
    let source_features = v2d![v1d![1.0, 2.0], v1d![3.0, 4.0]]; // 2 edges, 2 features each
    let target_features = v2d![v1d![3.0, 4.0], v1d![1.0, 2.0]]; // 2 edges, 2 features each

    let messages = mp.message(&source_features, &target_features, None);
    assert_eq!(messages.shape(), (2, 2)); // 2 edges, 2 features
    assert_eq!(messages[(0, 0)].data(), 1.0);
    assert_eq!(messages[(0, 1)].data(), 2.0);
    assert_eq!(messages[(1, 0)].data(), 3.0);
    assert_eq!(messages[(1, 1)].data(), 4.0);

    // Test full propagation
    let result = mp.propagate(&edge_index, &node_features, None);
    assert_eq!(result.shape(), (3, 2)); // 3 nodes, 2 features

    // Node 0: no incoming edges, should be zeros
    assert_eq!(result[(0, 0)].data(), 0.0);
    assert_eq!(result[(0, 1)].data(), 0.0);

    // Node 1: receives message from node 0
    assert_eq!(result[(1, 0)].data(), 1.0);
    assert_eq!(result[(1, 1)].data(), 2.0);

    // Node 2: receives message from node 1
    assert_eq!(result[(2, 0)].data(), 3.0);
    assert_eq!(result[(2, 1)].data(), 4.0);
}

#[test]
fn test_message_passing_propagate() {
    // Create a graph with 2 nodes and 1 edge: 0->1
    let edge_index = v2d![
        v1d![0.0], // source nodes
        v1d![1.0]  // target nodes
    ];

    let node_features = v2d![
        v1d![2.0, 3.0], // node 0
        v1d![4.0, 5.0]  // node 1
    ];

    let mp = MessagePassing::new();

    let result = mp.propagate(&edge_index, &node_features, None);
    assert_eq!(result.shape(), (2, 2));

    // Node 0: no incoming messages
    assert_eq!(result[(0, 0)].data(), 0.0);
    assert_eq!(result[(0, 1)].data(), 0.0);

    // Node 1: receives message from node 0
    assert_eq!(result[(1, 0)].data(), 2.0);
    assert_eq!(result[(1, 1)].data(), 3.0);
}

#[test]
fn test_gcn_creation() {
    let gcn = GCNConvLayer::new(3, 2, true); // 3 input features, 2 output features, with bias

    // Check that parameters exist
    let params = gcn.parameters();
    assert_eq!(params.len(), 3 * 2 + 2); // weight matrix (3x2) + bias (2)

    // Test GCN without bias
    let gcn_no_bias = GCNConvLayer::new(3, 2, false);
    let params_no_bias = gcn_no_bias.parameters();
    assert_eq!(params_no_bias.len(), 3 * 2); // only weight matrix (3x2)
}

#[test]
fn test_gcn_forward() {
    let mut gcn = GCNConvLayer::new(2, 2, true);

    // Set up a simple graph: 0->1
    let edge_index = v2d![v1d![0.0], v1d![1.0]];

    // Set simple weights for predictable output
    gcn.set_weights(Value::from(1.0)); // Sets all weights to 1.0

    // Set zero bias
    gcn.set_bias(Value::from(0.0));

    // Input features
    let input = v2d![
        v1d![2.0, 3.0], // node 0
        v1d![4.0, 5.0]  // node 1
    ];

    let output = gcn.forward(&edge_index, &input);
    assert_eq!(output.shape(), (2, 2));

    // With weights set to 1.0, the weight matrix becomes [[1,1],[1,1]]
    // So input transformation: [2,3] -> [5,5], [4,5] -> [9,9]
    // With self-loops added, edges become: 0->0, 0->1, 1->1
    // Degrees: node 0 = 1 (self), node 1 = 2 (self + from 0)

    // Node 0 receives: only from self: [5,5] * (1/sqrt(1)) * (1/sqrt(1)) = [5,5] * 1.0
    let expected_0_val = 5.0;

    // Node 1 receives: from 0: [5,5] * (1/sqrt(1)) * (1/sqrt(2)) + from self: [9,9] * (1/sqrt(2)) * (1/sqrt(2))
    let norm_from_0 = 1.0 / 2.0_f64.sqrt();
    let norm_from_self = 1.0 / 2.0;
    let expected_1_val = 5.0 * norm_from_0 + 9.0 * norm_from_self;

    assert!((output[(0, 0)].data() - expected_0_val).abs() < 1e-10);
    assert!((output[(0, 1)].data() - expected_0_val).abs() < 1e-10);
    assert!((output[(1, 0)].data() - expected_1_val).abs() < 1e-10);
    assert!((output[(1, 1)].data() - expected_1_val).abs() < 1e-10);

    let mut gcn = GCNConvLayer::new(2, 2, true); // 2 input features -> 2 output features
    gcn.set_weights(Value::from(0.1));
    gcn.set_bias(Value::from(1.0));
    let x = v2d![
        v1d![0.7420, 0.7937],
        v1d![0.5608, 0.9991],
        v1d![0.7246, 0.6131],
        v1d![0.4161, 0.0920],
        v1d![0.8763, 0.5896],
        v1d![0.3830, 0.1272],
        v1d![0.2332, 0.1119],
        v1d![0.1318, 0.5360],
        v1d![0.1574, 0.2773],
        v1d![0.1836, 0.3561]
    ];
    let edge_index = v2d![
        v1d![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 2, 0],
        v1d![1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 2, 0, 0]
    ];
    let expected_out = v2d![
        v1d![1.1178, 1.1178],
        v1d![1.1407, 1.1407],
        v1d![1.1595, 1.1595],
        v1d![1.0800, 1.0800],
        v1d![1.0987, 1.0987],
        v1d![1.0988, 1.0988],
        v1d![1.0428, 1.0428],
        v1d![1.0506, 1.0506],
        v1d![1.0551, 1.0551],
        v1d![1.0487, 1.0487]
    ];
    let out = gcn.forward(&edge_index, &x);
    assert_eq!(format!("{}", expected_out), format!("{}", out));
}

#[test]
fn test_sage_creation() {
    let sage = SAGEConvLayer::new(3, 2, true); // 3 input features, 2 output features, with bias

    // Check that parameters exist
    let params = sage.parameters();
    assert_eq!(params.len(), 3 * 2 + 3 * 2 + 2); // self_weight (3x2) + neighbor_weight (3x2) + bias (2)

    // Test SAGE without bias
    let sage_no_bias = SAGEConvLayer::new(3, 2, false);
    let params_no_bias = sage_no_bias.parameters();
    assert_eq!(params_no_bias.len(), 3 * 2 + 3 * 2); // only weight matrices
}

#[test]
fn test_sage_forward() {
    let mut sage = SAGEConvLayer::new(2, 2, true);

    // Set up a simple graph: 0->1
    let edge_index = v2d![v1d![0.0], v1d![1.0]];

    // Set identity weights for predictable output
    sage.set_root_weights(Value::from(1.0));
    sage.set_weights(Value::from(1.0));

    // Set zero bias
    sage.set_bias(Value::from(0.0));

    // Input features
    let input = v2d![
        v1d![2.0, 3.0], // node 0
        v1d![4.0, 5.0]  // node 1
    ];

    let output = sage.forward(&edge_index, &input);
    assert_eq!(output.shape(), (2, 2));

    // With weights set to 1.0, both weight matrices become [[1,1],[1,1]]
    // Self features: [2,3] -> [5,5], [4,5] -> [9,9]
    // Neighbor aggregation: Node 0 gets no neighbors (0,0), Node 1 gets [2,3] -> [5,5]
    // Combined: Node 0: [5,5] + [0,0] = [5,5], Node 1: [9,9] + [5,5] = [14,14]

    // Node 0: self features (5,5) + no neighbors = (5,5)
    assert_eq!(output[(0, 0)].data(), 5.0);
    assert_eq!(output[(0, 1)].data(), 5.0);

    // Node 1: self features (9,9) + neighbor features from node 0 (5,5) = (14,14)
    assert_eq!(output[(1, 0)].data(), 14.0);
    assert_eq!(output[(1, 1)].data(), 14.0);

    let mut sage = SAGEConvLayer::new(2, 2, true); // 2 input features -> 2 output features
    sage.set_weights(Value::from(0.1));
    sage.set_root_weights(Value::from(0.1));
    sage.set_bias(Value::from(1.0));
    let x = v2d![
        v1d![0.7420, 0.7937],
        v1d![0.5608, 0.9991],
        v1d![0.7246, 0.6131],
        v1d![0.4161, 0.0920],
        v1d![0.8763, 0.5896],
        v1d![0.3830, 0.1272],
        v1d![0.2332, 0.1119],
        v1d![0.1318, 0.5360],
        v1d![0.1574, 0.2773],
        v1d![0.1836, 0.3561]
    ];
    let edge_index = v2d![
        v1d![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 2, 0],
        v1d![1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 2, 0, 0]
    ];
    let expected_out = v2d![
        v1d![1.2673, 1.2673],
        v1d![1.3096, 1.3096],
        v1d![1.2885, 1.2885],
        v1d![1.1846, 1.1846],
        v1d![1.1974, 1.1974],
        v1d![1.1976, 1.1976],
        v1d![1.0855, 1.0855],
        v1d![1.1013, 1.1013],
        v1d![1.1102, 1.1102],
        v1d![1.0974, 1.0974]
    ];
    let out = sage.forward(&edge_index, &x);
    assert_eq!(format!("{}", expected_out), format!("{}", out));
}

#[test]
#[should_panic(expected = "edge_index must have shape (2, num_edges)")]
fn test_message_passing_without_edge_index() {
    let mp = MessagePassing::new();
    let dummy_features = v2d![v1d![1.0, 2.0]];
    // Create invalid edge index with wrong shape
    let invalid_edge_index = v2d![v1d![0.0]]; // Wrong shape: (1, 1) instead of (2, num_edges)
    mp.propagate(&invalid_edge_index, &dummy_features, None); // Should panic
}

#[test]
#[should_panic(expected = "edge_index must have shape (2, num_edges)")]
fn test_invalid_edge_index_shape() {
    let mp = MessagePassing::new();
    let dummy_features = v2d![v1d![1.0, 2.0]];
    let invalid_edge_index = v2d![
        v1d![0.0, 1.0],
        v1d![1.0, 2.0],
        v1d![2.0, 3.0] // Wrong shape: (3, 2) instead of (2, num_edges)
    ];
    mp.propagate(&invalid_edge_index, &dummy_features, None); // Should panic
}

#[test]
fn test_multi_edge_aggregation() {
    // Test a node that receives messages from multiple neighbors
    // Graph: 0->2, 1->2 (node 2 receives from both 0 and 1)
    let edge_index = v2d![
        v1d![0.0, 1.0], // source nodes
        v1d![2.0, 2.0]  // target nodes
    ];

    let node_features = v2d![
        v1d![1.0, 2.0], // node 0
        v1d![3.0, 4.0], // node 1
        v1d![5.0, 6.0]  // node 2
    ];

    let mp = MessagePassing::new();

    let result = mp.propagate(&edge_index, &node_features, None);

    // Node 2 should receive sum of messages from nodes 0 and 1 (no normalization in base MessagePassing)
    // Node 2 receives: node_0_features + node_1_features = [1.0, 2.0] + [3.0, 4.0] = [4.0, 6.0]
    assert_eq!(result[(2, 0)].data(), 4.0);
    assert_eq!(result[(2, 1)].data(), 6.0);

    // Nodes 0 and 1 should have no incoming messages
    assert_eq!(result[(0, 0)].data(), 0.0);
    assert_eq!(result[(0, 1)].data(), 0.0);
    assert_eq!(result[(1, 0)].data(), 0.0);
    assert_eq!(result[(1, 1)].data(), 0.0);
}

#[test]
fn test_gcn_normalization() {
    // Test that GCN properly applies degree normalization
    // Graph: 0->2, 1->2 (node 2 receives from both 0 and 1)
    let edge_index = v2d![
        v1d![0.0, 1.0], // source nodes
        v1d![2.0, 2.0]  // target nodes
    ];

    let mut gcn = GCNConvLayer::new(2, 2, false); // 2 input features, 2 output features, no bias

    // Set identity weights for predictable output
    gcn.set_weights(Value::from(1.0)); // Sets all weights to 1.0

    let input = v2d![
        v1d![1.0, 2.0], // node 0
        v1d![3.0, 4.0], // node 1
        v1d![5.0, 6.0]  // node 2
    ];

    let output = gcn.forward(&edge_index, &input);

    // With weights set to 1.0, input transformation:
    // [1,2] -> [3,3], [3,4] -> [7,7], [5,6] -> [11,11]
    // With self-loops added, the graph becomes:
    // 0->0, 0->2, 1->1, 1->2, 2->2
    // Degrees: node 0 = 1 (self), node 1 = 1 (self), node 2 = 3 (self + 2 incoming)

    // Node 0: receives only its own features with self-loop
    // Transformed features [3,3], degree = 1, normalization = 1/sqrt(1) * 1/sqrt(1) = 1.0
    // Expected: [3,3] * 1.0 = [3,3]
    assert!((output[(0, 0)].data() - 3.0).abs() < 1e-10);
    assert!((output[(0, 1)].data() - 3.0).abs() < 1e-10);

    // Node 1: receives only its own features with self-loop
    // Transformed features [7,7], degree = 1, normalization = 1/sqrt(1) * 1/sqrt(1) = 1.0
    // Expected: [7,7] * 1.0 = [7,7]
    assert!((output[(1, 0)].data() - 7.0).abs() < 1e-10);
    assert!((output[(1, 1)].data() - 7.0).abs() < 1e-10);

    // Node 2: receives from nodes 0, 1, and itself
    // From node 0: [3,3] * (1/sqrt(1)) * (1/sqrt(3))
    // From node 1: [7,7] * (1/sqrt(1)) * (1/sqrt(3))
    // From itself: [11,11] * (1/sqrt(3)) * (1/sqrt(3))
    let norm_from_0_1 = 1.0 / 3.0_f64.sqrt();
    let norm_from_self = 1.0 / 3.0;

    let expected_2_0 = 3.0 * norm_from_0_1 + 7.0 * norm_from_0_1 + 11.0 * norm_from_self;
    let expected_2_1 = 3.0 * norm_from_0_1 + 7.0 * norm_from_0_1 + 11.0 * norm_from_self;

    assert!((output[(2, 0)].data() - expected_2_0).abs() < 1e-10);
    assert!((output[(2, 1)].data() - expected_2_1).abs() < 1e-10);
}

#[test]
fn test_gcn_bias_addition_order() {
    // Create edge index where node 2 receives messages from nodes 0 and 1
    let edge_index = v2d![
        v1d![0.0, 1.0], // source nodes
        v1d![2.0, 2.0]  // target nodes
    ];

    let mut gcn = GCNConvLayer::new(2, 2, true); // 2 input features, 2 output features, WITH bias

    // Set identity weights for predictable output
    gcn.set_weights(Value::from(1.0)); // Sets all weights to 1.0

    // Set bias values
    gcn.set_bias(Value::from(10.0)); // Sets all bias values to 10.0

    let input = v2d![
        v1d![1.0, 2.0], // node 0
        v1d![3.0, 4.0], // node 1
        v1d![5.0, 6.0]  // node 2
    ];

    let output = gcn.forward(&edge_index, &input);

    // With weights set to 1.0 and bias set to 10.0:
    // Input transformation: [1,2] -> [3,3], [3,4] -> [7,7], [5,6] -> [11,11]
    // Degrees: node 0 = 1 (self), node 1 = 1 (self), node 2 = 3 (self + 2 incoming)
    // From node 0: [3,3] * (1/sqrt(1)) * (1/sqrt(3))
    // From node 1: [7,7] * (1/sqrt(1)) * (1/sqrt(3))
    // From itself: [11,11] * (1/sqrt(3)) * (1/sqrt(3))

    let norm_from_0_1 = 1.0 / 3.0_f64.sqrt();
    let norm_from_self = 1.0 / 3.0;

    let aggregated_value = 3.0 * norm_from_0_1 + 7.0 * norm_from_0_1 + 11.0 * norm_from_self;
    let expected_with_bias = aggregated_value + 10.0;

    assert!(
        (output[(2, 0)].data() - expected_with_bias).abs() < 1e-10,
        "Expected {}, got {}",
        expected_with_bias,
        output[(2, 0)].data()
    );
    assert!(
        (output[(2, 1)].data() - expected_with_bias).abs() < 1e-10,
        "Expected {}, got {}",
        expected_with_bias,
        output[(2, 1)].data()
    );

    // Node 0: receives [3,3] * 1.0 + bias = [3,3] + [10, 10] = [13, 13]
    // Node 1: receives [7,7] * 1.0 + bias = [7,7] + [10, 10] = [17, 17]
    let expected_0_with_bias = 3.0 + 10.0;
    let expected_1_with_bias = 7.0 + 10.0;

    assert!((output[(0, 0)].data() - expected_0_with_bias).abs() < 1e-10);
    assert!((output[(0, 1)].data() - expected_0_with_bias).abs() < 1e-10);
    assert!((output[(1, 0)].data() - expected_1_with_bias).abs() < 1e-10);
    assert!((output[(1, 1)].data() - expected_1_with_bias).abs() < 1e-10);
}

#[test]
fn test_gcn_multi_layer() {
    // Create a simple graph with 3 nodes and some edges
    let edge_index = v2d![
        v1d![0.0, 1.0, 2.0], // source nodes
        v1d![1.0, 2.0, 0.0]  // target nodes
    ];

    // Node features: 3 nodes, 4 features each
    let x = v2d![
        v1d![1.0, 2.0, 3.0, 4.0],    // node 0
        v1d![5.0, 6.0, 7.0, 8.0],    // node 1
        v1d![9.0, 10.0, 11.0, 12.0]  // node 2
    ];

    // Test multi-layer GCN: 4 -> 6 -> 2 (2 layers)
    let gcn = GCN::new(4, 2, 6, 0.0, 2);

    let output = gcn.forward(&edge_index, &x);

    // Should have 3 nodes with 2 output features each
    assert_eq!(output.shape(), (3, 2));

    // Verify output is reasonable (non-zero)
    for i in 0..3 {
        for j in 0..2 {
            assert_ne!(
                output[(i, j)].data(),
                0.0,
                "Output should not be zero at ({}, {})",
                i,
                j
            );
        }
    }
}

#[test]
fn test_sage_multi_layer() {
    // Create a simple graph with 3 nodes and some edges
    let edge_index = v2d![
        v1d![0.0, 1.0, 2.0], // source nodes
        v1d![1.0, 2.0, 0.0]  // target nodes
    ];

    // Node features: 3 nodes, 3 features each
    let x = v2d![
        v1d![1.0, 2.0, 3.0], // node 0
        v1d![4.0, 5.0, 6.0], // node 1
        v1d![7.0, 8.0, 9.0]  // node 2
    ];

    // Test multi-layer SAGE: 3 -> 5 -> 2 (2 layers)
    let sage = SAGE::new(3, 2, 5, 0.0, 2);

    let output = sage.forward(&edge_index, &x);

    // Should have 3 nodes with 2 output features each
    assert_eq!(output.shape(), (3, 2));

    // Verify output is reasonable (non-zero)
    for i in 0..3 {
        for j in 0..2 {
            assert_ne!(
                output[(i, j)].data(),
                0.0,
                "Output should not be zero at ({}, {})",
                i,
                j
            );
        }
    }
}

#[test]
fn test_gcn_single_vs_multi_layer() {
    // Create a simple graph
    let edge_index = v2d![
        v1d![0.0, 1.0], // source nodes
        v1d![1.0, 0.0]  // target nodes
    ];

    // Node features: 2 nodes, 3 features each
    let x = v2d![
        v1d![1.0, 2.0, 3.0], // node 0
        v1d![4.0, 5.0, 6.0]  // node 1
    ];

    // Compare single layer created with new()
    // Note: current implementation uses hidden_features even for single layer
    let gcn_single = GCN::new(3, 4, 8, 0.0, 1); // 3->8->4 (single layer means 1 hidden layer + 1 final)
    let gcn_multi = GCN::new(3, 4, 8, 0.0, 1); // same as single layer

    let output_single = gcn_single.forward(&edge_index, &x);
    let output_multi = gcn_multi.forward(&edge_index, &x);

    // Both should have same output shape
    assert_eq!(output_single.shape(), output_multi.shape());
    assert_eq!(output_single.shape(), (2, 4));
}

#[test]
fn test_sage_single_vs_multi_layer() {
    // Create a simple graph
    let edge_index = v2d![
        v1d![0.0, 1.0], // source nodes
        v1d![1.0, 0.0]  // target nodes
    ];

    // Node features: 2 nodes, 3 features each
    let x = v2d![
        v1d![1.0, 2.0, 3.0], // node 0
        v1d![4.0, 5.0, 6.0]  // node 1
    ];

    // Compare single layer created with new()
    // Note: current implementation uses hidden_features even for single layer
    let sage_single = SAGE::new(3, 4, 8, 0.0, 1); // 3->8->4 (single layer means 1 hidden layer + 1 final)
    let sage_multi = SAGE::new(3, 4, 8, 0.0, 1); // same as single layer

    let output_single = sage_single.forward(&edge_index, &x);
    let output_multi = sage_multi.forward(&edge_index, &x);

    // Both should have same output shape
    assert_eq!(output_single.shape(), output_multi.shape());
    assert_eq!(output_single.shape(), (2, 4));
}

#[test]
fn test_new() {
    let edge_index = v2d![
        v1d![0.0, 1.0, 2.0], // source nodes
        v1d![1.0, 2.0, 0.0]  // target nodes
    ];
    // Node features: 3 nodes, 4 features each
    let x = v2d![
        v1d![1.0, 2.0, 3.0, 4.0],    // node 0
        v1d![5.0, 6.0, 7.0, 8.0],    // node 1
        v1d![9.0, 10.0, 11.0, 12.0]  // node 2
    ];

    // Single layer networks
    // Note: current implementation uses hidden_features even for single layer
    let gcn_single = GCN::new(4, 2, 6, 0.0, 1); // in=4, hidden=6, out=2, layers=1 (creates 4->6->2)
    let sage_single = SAGE::new(4, 2, 6, 0.0, 1);

    let gcn_output = gcn_single.forward(&edge_index, &x);
    let sage_output = sage_single.forward(&edge_index, &x);

    assert_eq!(gcn_output.shape(), (3, 2));
    assert_eq!(sage_output.shape(), (3, 2));

    // Two-layer networks
    let gcn_2layer = GCN::new(4, 2, 8, 0.0, 2); // in=4, out=2, hidden=8, dropout=0, layers=2
    let sage_2layer = SAGE::new(4, 2, 8, 0.0, 2);

    let gcn_2layer_output = gcn_2layer.forward(&edge_index, &x);
    let sage_2layer_output = sage_2layer.forward(&edge_index, &x);

    assert_eq!(gcn_2layer_output.shape(), (3, 2));
    assert_eq!(sage_2layer_output.shape(), (3, 2));

    // Three-layer networks
    let gcn_3layer = GCN::new(4, 2, 6, 0.0, 3); // in=4, out=2, hidden=6, dropout=0, layers=3
    let sage_3layer = SAGE::new(4, 2, 6, 0.0, 3);

    let gcn_3layer_output = gcn_3layer.forward(&edge_index, &x);
    let sage_3layer_output = sage_3layer.forward(&edge_index, &x);

    assert_eq!(gcn_3layer_output.shape(), (3, 2));
    assert_eq!(sage_3layer_output.shape(), (3, 2));

    // Verify that outputs are different (networks are different)
    for i in 0..3 {
        for j in 0..2 {
            assert_ne!(
                gcn_output[(i, j)].data(),
                gcn_2layer_output[(i, j)].data(),
                "Single and 2-layer GCN should produce different outputs"
            );
            assert_ne!(
                sage_output[(i, j)].data(),
                sage_2layer_output[(i, j)].data(),
                "Single and 2-layer SAGE should produce different outputs"
            );
        }
    }
}
