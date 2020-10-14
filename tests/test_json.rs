#![cfg(feature = "serde-1")]

use rusticsom::*;
mod data;

#[test]
fn t_full_test() {
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

    let json = map.to_json().unwrap();

    let mut map_imported = SOM::from_json(&json, None, None).unwrap();

    for x in data.genrows() {
        let r1 = map.winner(x);
        let r2 = map_imported.winner(x);
        assert_eq!(r1, r2);
    }
}
