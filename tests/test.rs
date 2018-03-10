extern crate rusticsom;
extern crate ndarray;

use rusticsom::*;
use ndarray::{Array1, Array2};

#[test]
fn t_test_som() {
    let mut map = SOM::create(2, 3, 5, false, Some(0.1), None, None, None);

    assert_eq!(map.winner(Array1::from(vec![0.5; 5])), (0, 0));

    let mut temp_train = Array2::<f64>::zeros((2, 5));
    for i in temp_train.iter_mut() {
        *i = 1.0;
    }

    map.train_batch(temp_train, 1);
    assert_eq!(map.winner(Array1::from(vec![0.5; 5])), (0, 1));
}

#[test]
fn t_test_size() {
    let map = SOM::create(2, 3, 5, false, Some(0.1), None, None, None);
    assert_eq!(map.get_size(), (2, 3));
}
#[test]
fn t_distance_map(){
    let mut map = SOM::create(2, 3, 5, false, Some(0.1), None, None, None);
    let mut temp_train = Array2::<f64>::zeros((2, 5));
    for i in temp_train.iter_mut() {
        *i = 1.0;
    }

    map.train_batch(temp_train, 1);
    let dist = map.distance_map();
    
    assert_ne!(dist[[0, 0]], 0.0);
    assert_ne!(dist[[1, 1]], 0.0);
}