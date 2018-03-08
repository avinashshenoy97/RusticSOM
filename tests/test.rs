extern crate RusticSOM;

use RusticSOM::*;

#[test]
fn t_create_som() {
    let map = SOM::create(2, 3, 5, true);

    assert_eq!(map.x, 2);
    assert_eq!(map.y, 3);
    assert_eq!(map.z, 5);
}