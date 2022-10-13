extern crate ndarray;
extern crate rusticsom;

use ndarray::array;
use rusticsom::*;

#[test]
fn t_full_test() {
    // Run with `cargo test -- --nocapture` to get output!
    // Plotted with Matplotlib
    let mut map = SOM::create(10, 10, 4, false, None, None, None, None);
    let data = array![
        [5.1, 3.5, 1.4, 0.2],
        [4.9, 3.0, 1.4, 0.2],
        [4.7, 3.2, 1.3, 0.2],
        [4.6, 3.1, 1.5, 0.2],
        [5.0, 3.6, 1.4, 0.2],
        [5.4, 3.9, 1.7, 0.4],
        [4.6, 3.4, 1.4, 0.3],
        [5.0, 3.4, 1.5, 0.2],
        [4.4, 2.9, 1.4, 0.2],
        [4.9, 3.1, 1.5, 0.1],
        [5.4, 3.7, 1.5, 0.2],
        [4.8, 3.4, 1.6, 0.2],
        [4.8, 3.0, 1.4, 0.1],
        [4.3, 3.0, 1.1, 0.1],
        [5.8, 4.0, 1.2, 0.2],
        [5.7, 4.4, 1.5, 0.4],
        [5.4, 3.9, 1.3, 0.4],
        [5.1, 3.5, 1.4, 0.3],
        [5.7, 3.8, 1.7, 0.3],
        [5.1, 3.8, 1.5, 0.3],
        [5.4, 3.4, 1.7, 0.2],
        [5.1, 3.7, 1.5, 0.4],
        [4.6, 3.6, 1.0, 0.2],
        [5.1, 3.3, 1.7, 0.5],
        [4.8, 3.4, 1.9, 0.2],
        [5.0, 3.0, 1.6, 0.2],
        [5.0, 3.4, 1.6, 0.4],
        [5.2, 3.5, 1.5, 0.2],
        [5.2, 3.4, 1.4, 0.2],
        [4.7, 3.2, 1.6, 0.2],
        [4.8, 3.1, 1.6, 0.2],
        [5.4, 3.4, 1.5, 0.4],
        [5.2, 4.1, 1.5, 0.1],
        [5.5, 4.2, 1.4, 0.2],
        [4.9, 3.1, 1.5, 0.1],
        [5.0, 3.2, 1.2, 0.2],
        [5.5, 3.5, 1.3, 0.2],
        [4.9, 3.1, 1.5, 0.1],
        [4.4, 3.0, 1.3, 0.2],
        [5.1, 3.4, 1.5, 0.2],
        [5.0, 3.5, 1.3, 0.3],
        [4.5, 2.3, 1.3, 0.3],
        [4.4, 3.2, 1.3, 0.2],
        [5.0, 3.5, 1.6, 0.6],
        [5.1, 3.8, 1.9, 0.4],
        [4.8, 3.0, 1.4, 0.3],
        [5.1, 3.8, 1.6, 0.2],
        [4.6, 3.2, 1.4, 0.2],
        [5.3, 3.7, 1.5, 0.2],
        [5.0, 3.3, 1.4, 0.2],
        [7.0, 3.2, 4.7, 1.4],
        [6.4, 3.2, 4.5, 1.5],
        [6.9, 3.1, 4.9, 1.5],
        [5.5, 2.3, 4.0, 1.3],
        [6.5, 2.8, 4.6, 1.5],
        [5.7, 2.8, 4.5, 1.3],
        [6.3, 3.3, 4.7, 1.6],
        [4.9, 2.4, 3.3, 1.0],
        [6.6, 2.9, 4.6, 1.3],
        [5.2, 2.7, 3.9, 1.4],
        [5.0, 2.0, 3.5, 1.0],
        [5.9, 3.0, 4.2, 1.5],
        [6.0, 2.2, 4.0, 1.0],
        [6.1, 2.9, 4.7, 1.4],
        [5.6, 2.9, 3.6, 1.3],
        [6.7, 3.1, 4.4, 1.4],
        [5.6, 3.0, 4.5, 1.5],
        [5.8, 2.7, 4.1, 1.0],
        [6.2, 2.2, 4.5, 1.5],
        [5.6, 2.5, 3.9, 1.1],
        [5.9, 3.2, 4.8, 1.8],
        [6.1, 2.8, 4.0, 1.3],
        [6.3, 2.5, 4.9, 1.5],
        [6.1, 2.8, 4.7, 1.2],
        [6.4, 2.9, 4.3, 1.3],
        [6.6, 3.0, 4.4, 1.4],
        [6.8, 2.8, 4.8, 1.4],
        [6.7, 3.0, 5.0, 1.7],
        [6.0, 2.9, 4.5, 1.5],
        [5.7, 2.6, 3.5, 1.0],
        [5.5, 2.4, 3.8, 1.1],
        [5.5, 2.4, 3.7, 1.0],
        [5.8, 2.7, 3.9, 1.2],
        [6.0, 2.7, 5.1, 1.6],
        [5.4, 3.0, 4.5, 1.5],
        [6.0, 3.4, 4.5, 1.6],
        [6.7, 3.1, 4.7, 1.5],
        [6.3, 2.3, 4.4, 1.3],
        [5.6, 3.0, 4.1, 1.3],
        [5.5, 2.5, 4.0, 1.3],
        [5.5, 2.6, 4.4, 1.2],
        [6.1, 3.0, 4.6, 1.4],
        [5.8, 2.6, 4.0, 1.2],
        [5.0, 2.3, 3.3, 1.0],
        [5.6, 2.7, 4.2, 1.3],
        [5.7, 3.0, 4.2, 1.2],
        [5.7, 2.9, 4.2, 1.3],
        [6.2, 2.9, 4.3, 1.3],
        [5.1, 2.5, 3.0, 1.1],
        [5.7, 2.8, 4.1, 1.3],
        [6.3, 3.3, 6.0, 2.5],
        [5.8, 2.7, 5.1, 1.9],
        [7.1, 3.0, 5.9, 2.1],
        [6.3, 2.9, 5.6, 1.8],
        [6.5, 3.0, 5.8, 2.2],
        [7.6, 3.0, 6.6, 2.1],
        [4.9, 2.5, 4.5, 1.7],
        [7.3, 2.9, 6.3, 1.8],
        [6.7, 2.5, 5.8, 1.8],
        [7.2, 3.6, 6.1, 2.5],
        [6.5, 3.2, 5.1, 2.0],
        [6.4, 2.7, 5.3, 1.9],
        [6.8, 3.0, 5.5, 2.1],
        [5.7, 2.5, 5.0, 2.0],
        [5.8, 2.8, 5.1, 2.4],
        [6.4, 3.2, 5.3, 2.3],
        [6.5, 3.0, 5.5, 1.8],
        [7.7, 3.8, 6.7, 2.2],
        [7.7, 2.6, 6.9, 2.3],
        [6.0, 2.2, 5.0, 1.5],
        [6.9, 3.2, 5.7, 2.3],
        [5.6, 2.8, 4.9, 2.0],
        [7.7, 2.8, 6.7, 2.0],
        [6.3, 2.7, 4.9, 1.8],
        [6.7, 3.3, 5.7, 2.1],
        [7.2, 3.2, 6.0, 1.8],
        [6.2, 2.8, 4.8, 1.8],
        [6.1, 3.0, 4.9, 1.8],
        [6.4, 2.8, 5.6, 2.1],
        [7.2, 3.0, 5.8, 1.6],
        [7.4, 2.8, 6.1, 1.9],
        [7.9, 3.8, 6.4, 2.0],
        [6.4, 2.8, 5.6, 2.2],
        [6.3, 2.8, 5.1, 1.5],
        [6.1, 2.6, 5.6, 1.4],
        [7.7, 3.0, 6.1, 2.3],
        [6.3, 3.4, 5.6, 2.4],
        [6.4, 3.1, 5.5, 1.8],
        [6.0, 3.0, 4.8, 1.8],
        [6.9, 3.1, 5.4, 2.1],
        [6.7, 3.1, 5.6, 2.4],
        [6.9, 3.1, 5.1, 2.3],
        [5.8, 2.7, 5.1, 1.9],
        [6.8, 3.2, 5.9, 2.3],
        [6.7, 3.3, 5.7, 2.5],
        [6.7, 3.0, 5.2, 2.3],
        [6.3, 2.5, 5.0, 1.9],
        [6.5, 3.0, 5.2, 2.0],
        [6.2, 3.4, 5.4, 2.3],
        [5.9, 3.0, 5.1, 1.8],
    ];

    let data2 = data.to_owned();
    map.train_random(data, 1000);

    let dist_map = map.distance_map();
    println!("{:?}", dist_map);

    for x in data2.genrows() {
        let y = x.to_owned();
        print!("{:?}, ", map.winner(y));
    }

    let json = map.to_json().unwrap();

    let mut map_imported = SOM::from_json(&json, None, None, None).unwrap();

    for x in data2.genrows() {
        let y1 = x.to_owned();
        let y2 = x.to_owned();
        let r1 = map.winner(y1);
        let r2 = map_imported.winner(y2);
        assert_eq!(r1, r2);
    }
}
