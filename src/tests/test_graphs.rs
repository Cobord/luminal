use rand::{rngs::StdRng, SeedableRng};

use crate::prelude::*;

use super::random_vec_rng;
#[cfg(test)]
use dfdx::prelude::*;

pub fn matmul() -> (Graph, Vec<GraphTensor>) {
    let mut rng = StdRng::seed_from_u64(0);
    let mut cx = Graph::new();
    let a = cx
        .tensor(('a', 3))
        .set_dyn(random_vec_rng(2 * 3, &mut rng), (2, 3));
    let b = cx.tensor((3, 3)).set(random_vec_rng(3 * 3, &mut rng));
    let c = a.matmul(b).retrieve();
    (cx, vec![c])
}

pub fn batch_matmul() -> (Graph, Vec<GraphTensor>) {
    let mut rng = StdRng::seed_from_u64(0);
    let mut cx = Graph::new();
    let a = cx
        .tensor(('a', 'b', 2))
        .set_dyn(random_vec_rng(2 * 3 * 2, &mut rng), (2, 3, 2));
    let b = cx.tensor((2, 4)).set(random_vec_rng(2 * 4, &mut rng));
    let c = a.matmul(b).retrieve();
    (cx, vec![c])
}

#[test]
fn execute_no_delete_keeps_tensors() {
    let mut cx = Graph::new();
    let a = cx.tensor(3).set([1., 2., 3.]);
    let b = cx.tensor(3).set([4., 5., 6.]);
    let c = (a + b).retrieve();

    // first run without deleting tensors
    cx.execute_no_delete();

    // ensure all tensors remain in the map
    assert!(cx.tensors.contains_key(&(a.id, 0)));
    assert!(cx.tensors.contains_key(&(b.id, 0)));
    assert!(cx.tensors.contains_key(&(c.id, 0)));

    // normal execute should still produce the correct result
    cx.execute();

    let d_dev = dfdx::tensor::Cpu::default();
    let d_a = d_dev.tensor([1., 2., 3.]);
    let d_b = d_dev.tensor([4., 5., 6.]);
    let d_c = d_a + d_b;

    super::assert_close(&c.data(), &d_c.as_vec());
}

#[test]
fn disjoint_pieces() {
    let mut cx0 = Graph::new();
    let a0 = cx0.tensor(3).set([1., 2., 3.]);
    let b0 = cx0.tensor(3).set([4., 5., 6.]);
    let _c0 = (a0 + b0).retrieve();
    let mut cx1 = Graph::new();
    let a1 = cx1.tensor(3).set([7., 8., 9.]);
    let b1 = cx1.tensor(3).set([8., 7., 8.]);
    let _c1 = (a1 * b1).retrieve();

    let mut cx = cx0.disjoint_union(cx1);

    cx.execute();

    let d_dev = dfdx::tensor::Cpu::default();
    let d_a = d_dev.tensor([1., 2., 3.]);
    let d_b = d_dev.tensor([4., 5., 6.]);
    let d_c0 = d_a + d_b;
    let d_a = d_dev.tensor([7., 8., 9.]);
    let d_b = d_dev.tensor([8., 7., 8.]);
    let d_c1 = d_a * d_b;

    let mut c0c1 = cx.to_retrieve_graph_tensors();
    let c0 = c0c1.next().unwrap();
    let c1 = c0c1.next().unwrap();
    assert_eq!(c0c1.count(), 0);
    super::assert_close(&c0.data(), &d_c0.as_vec());
    super::assert_close(&c1.data(), &d_c1.as_vec());
}
