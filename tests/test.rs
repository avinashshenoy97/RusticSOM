use approx::assert_relative_eq;
use ndarray::{Array2, ArrayView1};
use rusticsom::*;
mod data;

#[test]
fn t_test_som() {
    let mut map = SOM::create(2, 3, 5, false, Some(0.1), None, None, None);

    assert_eq!(map.winner(ArrayView1::from(&[0.5; 5])), (0, 0));

    let mut temp_train = Array2::<f64>::zeros((2, 5));
    for i in temp_train.iter_mut() {
        *i = 1.0;
    }

    map.train_batch(temp_train.view(), 1);
    assert_eq!(map.winner(ArrayView1::from(&[0.5; 5])), (0, 1));
}

#[test]
fn t_test_size() {
    let map = SOM::create(2, 3, 5, false, Some(0.1), None, None, None);
    assert_eq!(map.get_size(), (2, 3));
}

#[test]
fn t_distance_map() {
    let mut map = SOM::create(2, 3, 5, false, Some(0.1), None, None, None);
    let mut temp_train = Array2::<f64>::zeros((2, 5));
    for i in temp_train.iter_mut() {
        *i = 1.0;
    }

    map.train_batch(temp_train.view(), 1);
    let dist = map.distance_map();

    assert_ne!(dist[[0, 0]], 0.0);
    assert_ne!(dist[[1, 1]], 0.0);
}

#[test]
fn t_full_test_random() {
    // Run with `cargo test -- --nocapture` to get output!
    // Plotted with Matplotlib
    let mut map = SOM::create(10, 10, 4, false, None, None, None, None);
    let data = ndarray::arr2(&data::IRIS);

    map.train_random(data.view(), 1000);

    let dist_map = map.distance_map();
    println!("{:?}", dist_map);

    for x in data.genrows() {
        print!("{:?}, ", map.winner(x));
    }
}

#[test]
fn t_full_test_batch() {
    // Run with `cargo test -- --nocapture` to get output!
    // Plotted with Matplotlib
    let mut map = SOM::create(10, 10, 4, false, None, None, None, None);
    let data = ndarray::arr2(&data::IRIS);

    map.train_batch(data.view(), 1000);

    let dist_map = map.distance_map();
    assert_relative_eq!(dist_map, ndarray::arr2(&data::IRIS_BATCH_DISTANCES));
    println!("{:?}", dist_map);

    for x in data.genrows() {
        print!("{:?}, ", map.winner(x));
    }
}
