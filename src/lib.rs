extern crate rand;
extern crate ndarray;

use rand::random as random;
use rand::Rng;
use ndarray::{Array1, Array2, Array3, Axis, ArrayView1, ArrayView2};
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
    activation_map: Array2<usize>,              // each cell represents how many times the corresponding cell in SOM was winner
    decay_function: fn(f32, u32, u32) -> f64,          // the function used to decay learning_rate and sigma
    neighbourhood_function: fn((usize, usize), (usize, usize), f32) -> Array2<f64>,          // the function that determines the weights of the neighbours
}

// Method definitions of the SOM struct
impl SOM {

    // To create a Self-Organizing Map (SOM)
    pub fn create(length: usize, breadth: usize, inputs: usize, randomize: bool, learning_rate: Option<f32>, sigma: Option<f32>, decay_function: Option<fn(f32, u32, u32) -> f64>, neighbourhood_function: Option<fn((usize, usize), (usize, usize), f32) -> Array2<f64>>) -> SOM {
        // Map of "length" x "breadth" is created, with depth "inputs" (for input vectors accepted by this SOM)
        // randomize: boolean; whether the SOM must be initialized with random weights or not

        let mut the_map = Array3::<f64>::zeros((length, breadth, inputs));
        let act_map = Array2::<usize>::zeros((length, breadth));
        let mut _init_regulate_lrate = 0;

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
            activation_map: act_map,
            map: the_map,
            regulate_lrate: _init_regulate_lrate
        }
    }

    pub fn winner(&mut self, elem: Array1<f64>) -> (usize, usize) {
        let mut temp: Array1<f64> = Array1::<f64>::zeros((self.z));
        let mut min: f64 = std::f64::MAX;
        let mut _temp_norm: f64 = 0.0;
        let mut ret: (usize, usize) = (0, 0);

        for i in 0..self.x {
            for j in 0..self.y {
                for k in 0..self.z {
                    temp[k] = self.map[[i, j, k]] - elem[[k]];
                }

                _temp_norm = norm(temp.view());

                if _temp_norm < min {
                    min = _temp_norm;
                    ret = (i, j);
                }
            }
        }

        if let Some(elem) = self.activation_map.get_mut(ret) {
            *(elem) += 1;
        }

        ret
    }

    // Update the weights of the SOM
    fn update(&mut self, elem: Array1<f64>, winner: (usize, usize), iteration_index: u32) {
        let new_lr = (self.decay_function)(self.learning_rate, iteration_index, self.regulate_lrate);
        let new_sig = (self.decay_function)(self.sigma, iteration_index, self.regulate_lrate);

        let g = (self.neighbourhood_function)((self.x, self.y), winner, new_sig as f32) * new_lr;

        let mut _temp_norm: f64 = 0.0;
        
        for i in 0..self.x {
            for j in 0..self.y {
                for k in 0..self.z {
                    self.map[[i, j, k]] += (elem[[k]] - self.map[[i, j, k]]) * g[[i, j]];
                }

                _temp_norm = norm(self.map.subview(Axis(0), i).subview(Axis(0), j));
                for k in 0..self.z {
                    self.map[[i, j, k]] /= _temp_norm;
                }
            }
        }
    }

    // Trains the SOM by picking random data points as inputs from the dataset
    pub fn train_random(&mut self, data: Array2<f64>, iterations: u32){
        let mut random_value: i32;
        let mut temp1: Array1<f64>;
        let mut temp2: Array1<f64>;
        self.update_regulate_lrate(iterations);
        for iteration in 0..iterations{
            temp1 = Array1::<f64>::zeros((ndarray::ArrayBase::dim(&data).1));
            temp2 = Array1::<f64>::zeros((ndarray::ArrayBase::dim(&data).1));
            random_value = rand::thread_rng().gen_range(0, ndarray::ArrayBase::dim(&data).0 as i32);
            for i in 0..ndarray::ArrayBase::dim(&data).1 {
                temp1[i] = data[[random_value as usize, i]];
                temp2[i] = data[[random_value as usize, i]];
            }
            let mut win = self.winner(temp1);
            self.update(temp2, win, iteration);
        }
    }   

    // Trains the SOM by picking  data points in batches (sequentially) as inputs from the dataset
    pub fn train_batch(&mut self, data: Array2<f64>, iterations: u32){
        let mut index: u32;
        let mut temp1: Array1<f64>;
        let mut temp2: Array1<f64>;
        self.update_regulate_lrate(ndarray::ArrayBase::dim(&data).0 as u32 * iterations);
        for iteration in 0..iterations{
            temp1 = Array1::<f64>::zeros((ndarray::ArrayBase::dim(&data).1));
            temp2 = Array1::<f64>::zeros((ndarray::ArrayBase::dim(&data).1));
            index = iteration % (ndarray::ArrayBase::dim(&data).0 - 1) as u32;
            for i in 0..ndarray::ArrayBase::dim(&data).1 {
                temp1[i] = data[[index as usize, i]];
                temp2[i] = data[[index as usize, i]];
            }
            let mut win = self.winner(temp1);
            self.update(temp2, win, iteration);
        }
    }  

    // Update learning rate regulator (keep learning rate constant with increase in number of iterations)
    fn update_regulate_lrate(&mut self, iterations: u32){
        self.regulate_lrate = iterations / 2;
    }

    pub fn activation_response(&self) -> ArrayView2<usize> {
        self.activation_map.view()
    }


    pub fn winner_dist(&mut self, elem: Array1<f64>) -> ((usize, usize), f64) {
        let mut tempelem = Array1::<f64>::zeros(elem.len());
        
        for i in 0..elem.len() {
            if let Some(temp) = tempelem.get_mut(i) {
                *(temp) = elem[i];
            }
        }

        let temp = self.winner(elem);

        (temp, euclid_dist(self.map.subview(Axis(0), temp.0).subview(Axis(0), temp.1), tempelem.view()))
    }

    #[cfg(test)]
    pub fn set_map_cell(&mut self, pos: (usize, usize, usize), val: f64) {
        if let Some(elem) = self.map.get_mut(pos) {
             *(elem) = val;
        }
    }

    #[cfg(test)]
    pub fn get_map_cell(&self, pos: (usize, usize, usize)) -> f64 {
        if let Some(elem) = self.map.get(pos) {
             *(elem)
        }
        else {
            panic!("Invalid index!");
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

fn euclid_dist(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    if a.len() != b.len() {
        panic!("Both arrays must be of same length to find Euclidian distance!");
    }

    let mut dist: f64 = 0.0;

    for i in 0..a.len() {
        dist += (a[i] - b[i]).powf(2.0);
    }

    dist.powf(0.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_winner() {
        let mut map = SOM::create(2, 3, 5, false, Some(0.1), None, None, None);

        for k in 0..5 {
            map.set_map_cell((1, 1, k), 1.5);
        }

        for k in 0..5 {
            assert_eq!(map.get_map_cell((1, 1, k)), 1.5);
        }

        assert_eq!(map.winner(Array1::from(vec![1.5; 5])), (1, 1));
        assert_eq!(map.winner(Array1::from(vec![0.5; 5])), (0, 0));
    }

    #[test]
    fn test_euclid() {
        let a = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
        let b = Array1::from(vec![4.0, 5.0, 6.0, 7.0]);

        assert_eq!(euclid_dist(a.view(), b.view()), 6.0);
    }
}