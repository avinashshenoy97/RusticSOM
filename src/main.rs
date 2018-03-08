extern crate rand;
extern crate ndarray;

use rand::random as random;
use ndarray::{Array3, Axis};
use std::fmt;

pub struct SOM {
    x: usize,           // length of SOM
    y: usize,           // breadth of SOM
    z: usize,           // size of inputs
    map: Array3<f64>,   // the SOM itself
}

// Method definitions of the SOM struct
impl SOM {

    // To create a Self-Organizing Map (SOM)
    fn create(length: usize, breadth: usize, inputs: usize, randomize: bool) -> SOM {
        // Map of "length" x "breadth" is created, with depth "inputs" (for input vectors accepted by this SOM)
        // randomize: boolean; whether the SOM must be initialized with random weights or not

        let mut the_map = Array3::<f64>::zeros((length, breadth, inputs));

        if randomize {
            for element in the_map.iter_mut() {
                *element = random::<f64>();
            }
        }

        SOM {
            x: length,
            y: breadth,
            z: inputs,
            map: the_map,
        }
    }
}

// To enable SOM objects to be printed with "print" and it's family of formatted string printing functions
impl fmt::Display for SOM {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (mut i, mut j) = (0, 0);

        for vector in self.map.lanes(Axis(2)) {
            println!("[{}, {}] : {}", i, j, vector);

            j += 1;
            if j == self.y {
                j = 0;
                i += 1;
            }
        }

        write!(f, "\nSOM Shape = ({}, {})\nExpected input vectors of length = {}", self.x, self.y, self.z)
    }
}

// Temporary main function for testing, should be removed when converted to proper library!
fn main() {
    let map = SOM::create(2, 3, 5, true);
    println!("{}", map);
}