extern crate rusticsom;

use rusticsom::*;

#[test]
fn t_create_som() {
    let mut map = SOM::create(2, 3, 5, false, Some(0.1), None, None, None);
}