extern crate rand;
extern crate ndarray;

use rand::random as random;
use rand::Rng;
use ndarray::{Array1, Array2, Array3, Axis, ArrayView1};
use std::fmt;
use std::f64::consts::PI as PI;

pub struct SOM {
    x: usize,               // length of SOM
    y: usize,               // breadth of SOM
    z: usize,               // size of inputs
    learning_rate: f32,   // initial learning rate
    sigma: f32,           // spread of neighbourhood function, default = 1.0
    regulate_lrate: u32,    // Regulates the learning rate w.r.t the number of iterations
    map: Array3<f64>,       // the SOM itself
    activation_map: Array2<f64>,       // the activation map
    decay_function: fn(f32, u32, u32) -> f64,          // the function used to decay learning_rate and sigma
    neighbourhood_function: fn((usize, usize), (usize, usize), f32) -> Array2<f64>,          // the function that determines the weights of the neighbours
}

// Method definitions of the SOM struct
impl SOM {

    // To create a Self-Organizing Map (SOM)
    fn create(length: usize, breadth: usize, inputs: usize, randomize: bool, learning_rate: Option<f32>, sigma: Option<f32>, decay_function: Option<fn(f32, u32, u32) -> f64>, neighbourhood_function: Option<fn((usize, usize), (usize, usize), f32) -> Array2<f64>>) -> SOM {
        // Map of "length" x "breadth" is created, with depth "inputs" (for input vectors accepted by this SOM)
        // randomize: boolean; whether the SOM must be initialized with random weights or not

        let mut the_map = Array3::<f64>::zeros((length, breadth, inputs));
        let mut the_activation_map = Array2::<f64>::zeros((length, breadth));
        let mut init_regulate_lrate = 0;

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
            activation_map: the_activation_map,
            regulate_lrate: init_regulate_lrate
        }
    }

    fn winner(&self, elem: Array1<f64>) -> (usize, usize) {
        let mut temp: Array1<f64> = Array1::<f64>::zeros((self.z));
        let mut min: f64 = std::f64::MAX;
        let mut temp_norm: f64 = 0.0;
        let mut ret: (usize, usize) = (0, 0);

        for i in 0..self.x {
            for j in 0..self.y {
                for k in 0..self.z {
                    temp[k] = self.map[[i, j, k]] - elem[[k]];
                }

                temp_norm = norm(temp.view());

                if temp_norm < min {
                    min = temp_norm;
                    ret = (i, j);
                }
            }
        }

        ret
    }

    // Update the weights of the SOM
    fn update(&mut self, elem: Array1<f64>, winner: (usize, usize), iteration_index: u32){
        let mut val = (self.decay_function)(self.learning_rate, iteration_index, self.regulate_lrate);
        let mut lsig = (self.decay_function)(self.sigma, iteration_index, self.regulate_lrate);
        let mut g = (self.neighbourhood_function)((self.x, self.y), winner, self.sigma) * val;
        let mut new_elem: Array1<f64>;
        let mut fnorm: f64;
        for i in 0..ndarray::ArrayBase::dim(&g).0{
            for j in 0..ndarray::ArrayBase::dim(&g).1{
                let mut temp1 = self.map.subview_mut(Axis(0), i);
                let mut temp2 = temp1.subview_mut(Axis(0), j);
                /*new_elem = elem - temp2; 
                temp2 = temp2 + g[[i, j]] * new_elem;
                fnorm = norm(temp);
                temp2 = temp2 / norm;*/
            }
        }
    }

    // Trains the SOM by picking random data points as inputs from the dataset
    fn train_random(&mut self, mut data: Array2<f64>, iterations: u32){
        let mut random_value: i32;
        self.update_regulate_lrate(iterations);
        for iteration in 0..iterations{
            random_value = rand::thread_rng().gen_range(0, ndarray::ArrayBase::dim(&data).0 as i32);
            // call to update function
        }
    }   

    // Trains the SOM by picking  data points in batches (sequentially) as inputs from the dataset
    fn train_batch(&mut self, mut data: Array2<f64>, iterations: u32){
        let mut index: u32;
        self.update_regulate_lrate(ndarray::ArrayBase::dim(&data).0 as u32 * iterations);
        for iteration in 0..iterations{
            index = iteration % (ndarray::ArrayBase::dim(&data).0 - 1) as u32;
            // call to update function
        }
    }  

    // Update learning rate regulator (keep learning rate constant with increase in number of iterations)
    fn update_regulate_lrate(&mut self, iterations: u32){
        self.regulate_lrate = iterations / 2;
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

        write!(f, "\nSOM Shape = ({}, {})\nExpected input vectors of length = {}\nSOM learning rate regulator = {}", self.x, self.y, self.z, self.regulate_lrate)
    }
}

fn norm(a: ArrayView1<f64>) -> f64 {
    let mut ret: f64 = 0.0;
    
    for i in a.iter() {
        ret += i.powf(2.0);
    }

    ret.powf(0.5)
}

fn default_decay_function(val: f32, curr_iter: u32, max_iter: u32) -> f64 {
    (val as f64) / ((1 + (curr_iter/max_iter)) as f64)
}

// Default neighbourhood function: Gaussian function; returns a Gaussian centered in pos
fn gaussian(size: (usize, usize), pos: (usize, usize), sigma: f32) -> Array2<f64> {
    let mut ret = Array2::<f64>::zeros((size.0, size.1));
    let div = 2.0 * PI * sigma as f64 * sigma as f64;

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
    let mut map = SOM::create(2, 3, 5, false, Some(0.1), None, None, None);
    
    for k in 0..5 {
        map.map[[1, 1, k]] = 2.0;
        map.map[[0, 1, k]] = 1.5;
    }

    println!("{}", map);
    println!("{:?}", map.winner(Array1::from_elem(5, 1.5)));
}