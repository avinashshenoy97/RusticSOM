use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
use rand::random;
use rand::Rng;
#[cfg(feature = "serde-1")]
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::fmt;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-1", derive(Deserialize, Serialize))]
pub struct SomData {
    x: usize,                      // length of SOM
    y: usize,                      // breadth of SOM
    z: usize,                      // size of inputs
    learning_rate: f32,            // initial learning rate
    sigma: f32,                    // spread of neighbourhood function, default = 1.0
    regulate_lrate: u32,           // Regulates the learning rate w.r.t the number of iterations
    map: Array3<f64>,              // the SOM itself
    activation_map: Array2<usize>, // each cell represents how many times the corresponding cell in SOM was winner
}

/// A function for determining neighbours' weights.
pub type NeighbourhoodFn = fn((usize, usize), (usize, usize), f64) -> Array2<f64>;

/// A function for decaying `learning_rate` and `sigma`.
pub type DecayFn = fn(f32, u32, u32) -> f64;

#[derive(Debug, Clone)]
pub struct SOM {
    data: SomData,
    decay_fn: DecayFn,
    neighbourhood_fn: NeighbourhoodFn,
}

// Method definitions of the SOM struct
impl SOM {
    // To create a Self-Organizing Map (SOM)
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        length: usize,
        breadth: usize,
        inputs: usize,
        randomize: bool,
        learning_rate: Option<f32>,
        sigma: Option<f32>,
        decay_fn: Option<DecayFn>,
        neighbourhood_fn: Option<NeighbourhoodFn>,
    ) -> SOM {
        // Map of "length" x "breadth" is created, with depth "inputs" (for input vectors accepted by this SOM)
        // randomize: boolean; whether the SOM must be initialized with random weights or not
        let the_map = if randomize {
            Array3::from_shape_simple_fn((length, breadth, inputs), random)
        } else {
            Array3::zeros((length, breadth, inputs))
        };

        let act_map = Array2::zeros((length, breadth));

        let data = SomData {
            x: length,
            y: breadth,
            z: inputs,
            learning_rate: learning_rate.unwrap_or(0.5),
            sigma: sigma.unwrap_or(1.0),
            activation_map: act_map,
            map: the_map,
            regulate_lrate: 0,
        };
        SOM {
            data,
            decay_fn: decay_fn.unwrap_or(default_decay_fn),
            neighbourhood_fn: neighbourhood_fn.unwrap_or(gaussian),
        }
    }

    /// Returns the `(x, y)` position of the winning neuron for `elem`.
    pub fn winner(&mut self, elem: ArrayView1<f64>) -> (usize, usize) {
        let mut min: f64 = std::f64::MAX;
        let mut ret: (usize, usize) = (0, 0);

        for (i, row) in self.data.map.outer_iter().enumerate() {
            for (j, entry) in row.outer_iter().enumerate() {
                let norm = norm2(entry, elem);
                if norm < min {
                    min = norm;
                    ret = (i, j);
                }
            }
        }

        self.data.activation_map[ret] += 1;
        ret
    }

    // Update the weights of the SOM
    fn update(&mut self, elem: ArrayView1<f64>, winner: (usize, usize), iteration_index: u32) {
        let new_lr = (self.decay_fn)(
            self.data.learning_rate,
            iteration_index,
            self.data.regulate_lrate,
        );
        let new_sig = (self.decay_fn)(self.data.sigma, iteration_index, self.data.regulate_lrate);

        let g = (self.neighbourhood_fn)((self.data.x, self.data.y), winner, new_sig) * new_lr;

        for i in 0..self.data.x {
            for j in 0..self.data.y {
                for k in 0..self.data.z {
                    self.data.map[[i, j, k]] += (elem[[k]] - self.data.map[[i, j, k]]) * g[[i, j]];
                }

                let norm = norm(self.data.map.index_axis(Axis(0), i).index_axis(Axis(0), j));
                for k in 0..self.data.z {
                    self.data.map[[i, j, k]] /= norm;
                }
            }
        }
    }

    // Trains the SOM by picking random data points as inputs from the dataset
    pub fn train_random(&mut self, data: ArrayView2<f64>, iterations: u32) {
        let (rows, cols) = data.dim();
        let rnd_row = || rand::thread_rng().gen_range(0, rows);
        self.update_regulate_lrate(iterations);
        // TODO: temporary heap allocation undesirable, but not easily
        //       avoidable.
        let mut randomized_col = Array1::<f64>::zeros(cols);
        for iteration in 0..iterations {
            // TODO: Does the algorithm require us to operate over
            //       columns like this? It is not optimal from a
            //       cache-coherency POV.
            for data_col in data.axis_iter(Axis(1)) {
                ndarray::Zip::from(&mut randomized_col)
                    .apply(|rc_elem| *rc_elem = data_col[rnd_row()]);
            }
            let win = self.winner(randomized_col.view());
            self.update(randomized_col.view(), win, iteration);
        }
    }

    // Trains the SOM by picking  data points in batches (sequentially) as inputs from the dataset
    pub fn train_batch(&mut self, data: ArrayView2<f64>, iterations: u32) {
        self.update_regulate_lrate(data.nrows() as u32 * iterations);
        for iteration in 0..iterations {
            let index = iteration as usize % (data.nrows() - 1);
            let col = data.index_axis(Axis(0), index);
            let win = self.winner(col);
            self.update(col, win, iteration);
        }
    }

    // Update learning rate regulator (keep learning rate constant with increase in number of iterations)
    fn update_regulate_lrate(&mut self, iterations: u32) {
        self.data.regulate_lrate = iterations / 2;
    }

    // Returns the activation map of the SOM, where each cell at (i, j) represents how many times the cell at (i, j) in the SOM was picked a winner neuron.
    pub fn activation_response(&self) -> ArrayView2<usize> {
        self.data.activation_map.view()
    }

    /// Returns a tuple of the winner and its distance from `elem`.
    pub fn winner_dist(&mut self, elem: ArrayView1<f64>) -> ((usize, usize), f64) {
        let winner = self.winner(elem);
        let distance = euclid_dist(
            self.data
                .map
                .index_axis(Axis(0), winner.0)
                .index_axis(Axis(0), winner.1),
            elem,
        );
        (winner, distance)
    }

    /// Returns `self`'s size in `(rows, cols)`.
    pub fn get_size(&self) -> (usize, usize) {
        (self.data.x, self.data.y)
    }

    // Returns the distance map of each neuron / the normalised sum of a neuron to every other neuron in the map.
    pub fn distance_map(&self) -> Array2<f64> {
        let mut dist_map = Array2::<f64>::zeros((self.data.x, self.data.y));
        let mut temp_dist: f64;
        let mut max_dist: f64 = 0.0;
        for i in 0..self.data.x {
            for j in 0..self.data.y {
                temp_dist = 0.0;
                for k in 0..self.data.x {
                    for l in 0..self.data.y {
                        temp_dist += euclid_dist(
                            self.data.map.index_axis(Axis(0), i).index_axis(Axis(0), j),
                            self.data.map.index_axis(Axis(0), k).index_axis(Axis(0), l),
                        );
                    }
                }
                if temp_dist > max_dist {
                    max_dist = temp_dist;
                }
                dist_map[[i, j]] = temp_dist;
            }
        }

        // A bug in an earlier version of this commit led to all
        // division by 0.0, which silent results in NaN and not a
        // process signal. This was made by worse by integration tests
        // not comparing test results to reference results. Let's
        // assert here for the time being.
        debug_assert!(max_dist.abs() > 0.0);

        for i in 0..self.data.x {
            for j in 0..self.data.y {
                dist_map[[i, j]] /= max_dist;
            }
        }
        dist_map
    }

    // Unit testing functions for setting individual cell weights
    #[cfg(test)]
    pub fn set_map_cell(&mut self, (i, j, k): (usize, usize, usize), val: f64) {
        self.data.map[[i, j, k]] = val;
    }

    // Unit testing functions for getting individual cell weights
    #[cfg(test)]
    pub fn get_map_cell(&self, (i, j, k): (usize, usize, usize)) -> f64 {
        self.data.map[[i, j, k]]
    }
}

// To enable SOM objects to be printed with "print" and it's family of formatted string printing functions
impl fmt::Display for SOM {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (mut i, mut j) = (0, 0);

        for vector in self.data.map.lanes(Axis(2)) {
            println!("[{}, {}] : {}", i, j, vector);

            j += 1;
            if j == self.data.y {
                j = 0;
                i += 1;
            }
        }

        write!(f, "\nSOM Shape = ({}, {})\nExpected input vectors of length = {}\nSOM learning rate regulator = {}", self.data.x, self.data.y, self.data.z, self.data.regulate_lrate)
    }
}

#[cfg(feature = "serde-1")]
impl SOM {
    pub fn from_json(
        serialized: &str,
        decay_fn: Option<DecayFn>,
        neighbourhood_fn: Option<NeighbourhoodFn>,
    ) -> serde_json::Result<SOM> {
        let data: SomData = serde_json::from_str(&serialized)?;

        Ok(SOM {
            data,
            decay_fn: decay_fn.unwrap_or(default_decay_fn),
            neighbourhood_fn: neighbourhood_fn.unwrap_or(gaussian),
        })
    }

    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(&self.data)
    }
}

// Returns the 2-norm of a vector.
fn norm<'a, I>(iti: I) -> f64
where
    I: IntoIterator<Item = &'a f64>,
{
    iti.into_iter().map(|elem| elem.powi(2)).sum::<f64>().sqrt()
}

// Returns the 2-norm of a vector.
fn norm2<'l, 'r, L, R>(lhs: L, rhs: R) -> f64
where
    L: IntoIterator<Item = &'l f64>,
    R: IntoIterator<Item = &'r f64>,
{
    lhs.into_iter()
        .zip(rhs.into_iter())
        .map(|(l, r)| (l - r) * (l - r))
        .sum::<f64>()
        .sqrt()
}

// The default decay function for LR and Sigma
fn default_decay_fn(val: f32, curr_iter: u32, max_iter: u32) -> f64 {
    (val as f64) / ((1 + (curr_iter / max_iter)) as f64)
}

/// Default neighborhood function.
///
/// Returns a two-dimensional Gaussian distribution centered at `pos`.
fn gaussian(dims: (usize, usize), pos: (usize, usize), sigma: f64) -> Array2<f64> {
    let div = 2.0 * PI * sigma * sigma;
    debug_assert!(div.abs() > f64::EPSILON);

    let shape_fn = |(i, j)| {
        let x = ((i as f64 - pos.0 as f64) * (i as f64 - pos.0 as f64) / -div).exp();
        let y = ((j as f64 - pos.1 as f64) * (j as f64 - pos.1 as f64) / -div).exp();
        x * y
    };

    // This allocation is fine. Benchmarking shows that eliminating
    // this allocation by changing making `NeighbourhoodFn` lazy
    // (computing a single array element at a time instead of
    // returning the whole array) actually causes a performance
    // regression of greater than 12%.
    Array2::from_shape_fn(dims, shape_fn)
}

/// Returns the [Euclidean distance] between `a` and `b`.
///
/// [Euclidean distance]: https://en.wikipedia.org/wiki/Euclidean_distance
fn euclid_dist(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    debug_assert_eq!(a.len(), b.len());

    a.iter()
        .zip(b.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt()
}

// Unit-testing module - only compiled when "cargo test" is run!
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_winner() {
        let mut map = SOM::create(2, 3, 5, false, Some(0.1), None, None, None);

        for k in 0..5 {
            map.set_map_cell((1, 1, k), 1.5);
        }

        for k in 0..5 {
            assert_eq!(map.get_map_cell((1, 1, k)), 1.5);
        }

        assert_eq!(map.winner(ArrayView1::from(&vec![1.5; 5])), (1, 1));
        assert_eq!(map.winner(ArrayView1::from(&vec![0.5; 5])), (0, 0));
    }

    #[test]
    fn test_euclid() {
        let a = array![1.0, 2.0, 3.0, 4.0];
        let b = array![4.0, 5.0, 6.0, 7.0];

        assert_eq!(euclid_dist(a.view(), b.view()), 6.0);
    }
}
