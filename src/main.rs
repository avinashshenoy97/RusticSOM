extern crate rand;
extern crate ndarray;

use rand::random as random;
use ndarray::{Array2, Array3, Axis};
use std::fmt;
use std::f64::consts::PI as PI;

pub struct SOM {
    x: usize,               // length of SOM
    y: usize,               // breadth of SOM
    z: usize,               // size of inputs
    learning_rate: f32,   // initial learning rate
    sigma: f64,           // spread of neighbourhood function, default = 1.0
    map: Array3<f64>,       // the SOM itself
    decay_function: fn(f64, u64, u64) -> f64,          // the function used to decay learning_rate and sigma
    neighbourhood_function: fn((usize, usize), (usize, usize), f64) -> Array2<f64>,          // the function used to decay learning_rate and sigma
}

// Method definitions of the SOM struct
impl SOM {

    // To create a Self-Organizing Map (SOM)
    fn create(length: usize, breadth: usize, inputs: usize, randomize: bool, learning_rate: Option<f32>, sigma: Option<f64>, decay_function: Option<fn(f64, u64, u64) -> f64>, neighbourhood_function: Option<fn((usize, usize), (usize, usize), f64) -> Array2<f64>>) -> SOM {
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
            learning_rate: match learning_rate {
                None => 0.5,
                Some(value) => value,
            },
            sigma: match sigma {
                None => 1.0,
                Some(value) => value,
            },
            decay_function: match decay_function {
                None => default_decay_function,
                Some(foo) => foo,
            },
            neighbourhood_function: match neighbourhood_function {
                None => gaussian,
                Some(foo) => foo,
            },
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

fn default_decay_function(val: f64, curr_iter: u64, max_iter: u64) -> f64 {
    (val as f64) / ((1 + (curr_iter/max_iter)) as f64)
}

// Default neighbourhood function: Gaussian function; returns a Gaussian centered in pos
fn gaussian(size: (usize, usize), pos: (usize, usize), sigma: f64) -> Array2<f64> {
    let mut ret = Array2::<f64>::zeros((size.0, size.1));
    let div = 2.0 * PI * sigma * sigma;

    let mut x: Vec<f64> = Vec::new();
    let mut y: Vec<f64> = Vec::new();

    for i in 0..size.0 {
        x.push(i as f64);
        if let Some(elem) = x.get_mut(i) {
            *elem = -((*elem - (pos.0 as f64)).powf(2.0) / div);
            *elem = (*elem).exp();
        }
    }

    for i in 0..size.1 {
        y.push(i as f64);
        if let Some(elem) = y.get_mut(i) {
            *elem = -((*elem - (pos.1 as f64)).powf(2.0) / div);
            *elem = (*elem).exp();
        }
    }

    for i in 0..size.0 {
        for j in 0..size.1 {
            ret[[i, j]] = x[i] * y[j];
        }
    }

    ret
}

// Temporary main function for testing, should be removed when converted to proper library!
fn main() {
    let map = SOM::create(2, 3, 5, true, Some(0.1), None, None, None);
    println!("{}", map);
}