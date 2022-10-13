//! The RustiSOM crate provides a Rust implementation of Self Organizing Feature Maps (SOMs)
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
use rand::random;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::fmt;

/// Contains the data about the Self Organising Map
#[derive(Serialize, Deserialize, Debug)]
pub struct SomData {
    /// Length of the SOM
    x: usize,
    /// Breadth of the SOM
    y: usize,
    /// The number of elements in each neuron. Must equal the number of features in each sample of
    /// the input data
    z: usize,
    /// initial learning rate
    learning_rate: f32,
    /// spread of neighbourhood function, default = 1.0
    sigma: f32,
    /// Regulates the learning rate w.r.t the number of iterations
    regulate_lrate: u32,
    /// the SOM itself
    map: Array3<f64>,
    /// each cell represents how many times the corresponding cell in SOM was winner
    activation_map: Array2<usize>,
}

/// A function for determining neighbours' weights.
pub type NeighbourhoodFn = fn((usize, usize), (usize, usize), f32) -> Array2<f64>;

/// A function for decaying `learning_rate` and `sigma`.
pub type DecayFn = fn(f32, u32, u32) -> f64;

/// A callback type that takes in the number of iterations
pub type CallbackFn = fn(&mut SOM, u32);

/// Describes the Self Organizing Map itself and provides constructors for creating one.
pub struct SOM {
    data: SomData,
    decay_fn: DecayFn,
    neighbourhood_fn: NeighbourhoodFn,
    /// An array of callbacks which are called after every training iteration.
    callbacks: Vec<CallbackFn>,
}

/// Method definitions of the SOM struct
impl SOM {
    /// Creates a Self Organising Map
    /// * `length` - The length of the SOM.
    /// * `breadth` - The breadth of the SOM.
    /// * `inputs` - The depth of the SOM. Each of the cells will have this many neurons.
    /// * `randomize` - whether the SOM must be initialised with random weights or with all zeros.
    /// * `learning_rate` - The learning rate to use. Defaults to 0.5 if `None`.
    /// * `sigma` - The sigma value to use. Defaults to 1.0 if `None`
    /// * `decay_fn` - The decay function to use. If `None`, will default to:
    /// ```
    /// fn default_decay_fn(val: f32, curr_iter: u32, max_iter: u32) -> f64 {
    ///     (val as f64) / ((1 + (curr_iter / max_iter)) as f64)
    /// }
    /// ```
    /// * `neighbourhood_fn` - The neighbourhood function to use. If `None`, will default to a 2
    ///   dimensional gaussian centered around the relevant cell and with a standard deviation that
    ///   decays over time according to `decay_fn`.
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
            callbacks: vec![],
        }
    }

    /// Find and return the position of the winner neuron for a given input sample. This is called
    /// the Best Matching Unit (BMU) in the literature
    ///
    /// TODO: (breaking-change) switch `elem` to `ArrayView1`. See todo
    ///       for `Self::winner_dist()`.
    pub fn winner(&mut self, elem: Array1<f64>) -> (usize, usize) {
        let mut delta: Array1<f64> = Array1::<f64>::zeros(self.data.z);
        let mut min_magnitude = f64::MAX;
        let mut winning_neuron = (0, 0);

        for i in 0..self.data.x {
            for j in 0..self.data.y {
                // delta = &self.data.map.slice(s![i, j, ..]) - &elem.view();
                // TODO this assertion just checks that the above broadcasting slice doesn't change
                // anything. the below assertion can be removed if nothing breaks for a while
                for k in 0..self.data.z {
                    // assert_eq!(delta[k], self.data.map[[i, j, k]] - elem[[k]]);
                    delta[k] = self.data.map[[i, j, k]] - elem[[k]];
                }

                let magnitude = norm(delta.view());

                if magnitude < min_magnitude {
                    min_magnitude = magnitude;
                    winning_neuron = (i, j);
                }
            }
        }
        // Increment the winning neuron's activation
        if let Some(elem) = self.data.activation_map.get_mut(winning_neuron) {
            *elem += 1;
        }
        winning_neuron
    }

    /// Use the provided JSON string to create a SOM. The JSON string should have come from
    /// `SOM::to_json`.
    pub fn from_json(
        serialized: &str,
        decay_fn: Option<DecayFn>,
        neighbourhood_fn: Option<NeighbourhoodFn>,
        callbacks: Option<Vec<CallbackFn>>,
    ) -> serde_json::Result<SOM> {
        let data: SomData = serde_json::from_str(serialized)?;

        Ok(SOM {
            data,
            decay_fn: decay_fn.unwrap_or(default_decay_fn),
            neighbourhood_fn: neighbourhood_fn.unwrap_or(gaussian),
            callbacks: callbacks.unwrap_or(vec![]),
        })
    }
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(&self.data)
    }

    /// Update the weights of the SOM given a sample `elem`, the winning neuron for that sample
    /// `winner`, and the iteration number `iteration_index`.
    fn update(&mut self, elem: Array1<f64>, winner: (usize, usize), iteration_index: u32) {
        // Decay the learning rate based on the decay function
        let new_lr = (self.decay_fn)(
            self.data.learning_rate,
            iteration_index,
            self.data.regulate_lrate,
        );
        // Decay the value of sigma based on the decay function
        let new_sigma = (self.decay_fn)(self.data.sigma, iteration_index, self.data.regulate_lrate);

        let g =
            (self.neighbourhood_fn)((self.data.x, self.data.y), winner, new_sigma as f32) * new_lr;

        // Iterate over every neuron
        for i in 0..self.data.x {
            for j in 0..self.data.y {
                // Iterate over every value in the current neuron
                for k in 0..self.data.z {
                    // Get the difference between the the new neuron value and the old neuron value
                    let delta = elem[[k]] - self.data.map[[i, j, k]];
                    // Update the neuron to be the change multiplied by the result of the
                    // neighbourhood function
                    self.data.map[[i, j, k]] += g[[i, j]] * delta;
                }
                // Normalise all of the values of this neuron
                let norm = norm(self.data.map.index_axis(Axis(0), i).index_axis(Axis(0), j));
                for k in 0..self.data.z {
                    self.data.map[[i, j, k]] /= norm;
                }
            }
        }
    }

    /// Train the SOM by picking samples from the dataset at random for `iterations` number of
    /// iterations.
    pub fn train_random(&mut self, data: Array2<f64>, iterations: u32) {
        let mut random_value: i32;
        let mut sample: Array1<f64>;
        let mut sample_copy: Array1<f64>;
        self.update_regulate_lrate(iterations);
        for iteration in 0..iterations {
            sample = Array1::<f64>::zeros(ndarray::ArrayBase::dim(&data).1);
            sample_copy = Array1::<f64>::zeros(ndarray::ArrayBase::dim(&data).1);
            // Randomly choose an index from `data`
            random_value = rand::thread_rng().gen_range(0, ndarray::ArrayBase::dim(&data).0 as i32);
            // Iterate over all values in the randomly chosen sample and add them to an array
            for i in 0..ndarray::ArrayBase::dim(&data).1 {
                sample[i] = data[[random_value as usize, i]];
                sample_copy[i] = data[[random_value as usize, i]];
            }
            // Find the winner neuron for the randomly chosen sample
            let winner = self.winner(sample);
            // Update the map based on the randomly chosen sample's results
            self.update(sample_copy, winner, iteration);
            // Iterate over the callbacks and execute them so the user can hook into training
            for cb in self.callbacks.clone() {
                (cb)(self, iteration);
            }
        }
    }

    /// Trains the SOM by picking data points in batches (sequentially) as inputs from the dataset
    pub fn train_batch(&mut self, data: Array2<f64>, iterations: u32) {
        let mut index: u32;
        let mut temp1: Array1<f64>;
        let mut temp2: Array1<f64>;
        self.update_regulate_lrate(ndarray::ArrayBase::dim(&data).0 as u32 * iterations);
        for iteration in 0..iterations {
            temp1 = Array1::<f64>::zeros(ndarray::ArrayBase::dim(&data).1);
            temp2 = Array1::<f64>::zeros(ndarray::ArrayBase::dim(&data).1);
            index = iteration % (ndarray::ArrayBase::dim(&data).0 - 1) as u32;
            for i in 0..ndarray::ArrayBase::dim(&data).1 {
                temp1[i] = data[[index as usize, i]];
                temp2[i] = data[[index as usize, i]];
            }
            let win = self.winner(temp1);
            self.update(temp2, win, iteration);
        }
    }

    /// Update learning rate regulator (keep learning rate constant with increase in number of
    /// iterations). This will set the learning rate regulator to be half of the provided number of
    /// iterations.
    fn update_regulate_lrate(&mut self, iterations: u32) {
        self.data.regulate_lrate = iterations / 2;
    }

    /// Returns the activation map of the SOM, where each cell at (i, j) represents how many times
    /// the cell at (i, j) in the SOM was picked a winner neuron.
    pub fn activation_response(&self) -> ArrayView2<usize> {
        self.data.activation_map.view()
    }

    /// Return both the coordinates of the winning neuron for a given sample, as well as the
    /// distance from that sample to the neuron. Similar to `SOM::winner()`. The winner is called
    /// the Best Matching Unit (BMU) in the literature
    ///
    /// TODO: (breaking-change) make `elem` an `ArrayView1` to remove
    ///       at least one heap allocation. Requires same change to
    ///       `Self::winner()`.
    pub fn winner_dist(&mut self, elem: Array1<f64>) -> ((usize, usize), f64) {
        let sample = elem.clone();
        let winning_node = self.winner(elem);

        (
            winning_node,
            euclid_dist(
                self.data
                    .map
                    .index_axis(Axis(0), winning_node.0)
                    .index_axis(Axis(0), winning_node.1),
                sample.view(),
            ),
        )
    }

    /// Get the dimensions of the SOM as a tuple `(length, breadth)`.
    pub fn get_size(&self) -> (usize, usize) {
        (self.data.x, self.data.y)
    }

    /// Get the normalised distance from every neuron to every other neuron (AKA the distance map).
    /// The result is a 2D array of size `(width, breadth)` where the value at `[i,j]` is the sum
    /// of the distances from neuron `i,j` to every other neuron. These values are standardised to
    /// have a maximum of `1.0`, and the minimum is at least `0.0`.
    ///
    /// Computed in O((`w` `b`)^2), where `w` is the width and `b` the breadth of the SOM.
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
        // Now normalise the distances by dividing by the largest distance.
        for i in 0..self.data.x {
            for j in 0..self.data.y {
                dist_map[[i, j]] /= max_dist;
            }
        }
        dist_map
    }

    /// Add a callback to the SOM. This function will be called after every training iteration and
    /// can be used to monitor progress or store the state of the SOM at checkpoints. For example:
    /// ```
    /// use rusticsom::SOM;
    /// use std::fs::File;
    /// use std::io::Write;
    ///
    /// let mut som = SOM::create(10, 10, 4, false, None, None, None, None);
    /// som.add_callback(|som, iters| {
    ///     if iters % 100 == 0 {
    ///         // Print out the number of passed iterations
    ///         println!("Iteration {}", iters);
    ///         // And save the som as a JSON
    ///         let mut file = File::create(format!("som_at_iter_{}.json", iters)).unwrap();
    ///         writeln!(&mut file, "{}", som.to_json().unwrap()).unwrap();
    ///     }
    /// });
    /// ```
    pub fn add_callback(&mut self, callback: CallbackFn) {
        self.callbacks.push(callback);
    }

    /// Unit testing functions for setting individual cell weights
    #[cfg(test)]
    pub fn set_map_cell(&mut self, (i, j, k): (usize, usize, usize), val: f64) {
        self.data.map[[i, j, k]] = val;
    }

    /// Unit testing functions for getting individual cell weights
    #[cfg(test)]
    pub fn get_map_cell(&self, (i, j, k): (usize, usize, usize)) -> f64 {
        self.data.map[[i, j, k]]
    }
}

/// Allow the SOM to be printed.
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

/// Returns the 2-norm of a vector represented as a 1D ArrayView. This is the square root of the
/// sum of the squares of all values in `a`.
fn norm(a: ArrayView1<f64>) -> f64 {
    a.iter().map(|elem| elem.powi(2)).sum::<f64>().sqrt()
}

/// The default decay function for learning rate and sigma. This will decay as the `curr_iter`
/// increases.
fn default_decay_fn(val: f32, curr_iter: u32, max_iter: u32) -> f64 {
    (val as f64) / ((1 + (curr_iter / max_iter)) as f64)
}

/// Returns a two-dimensional Gaussian distribution centered at `pos` of shape `dims` with a
/// standard deviation of `sigma`.
///
/// This is used as the default `neighbourhood_fn`.
fn gaussian(dims: (usize, usize), pos: (usize, usize), sigma: f32) -> Array2<f64> {
    let div = 2.0 * PI * (sigma as f64).powi(2);

    let shape_fn = |(i, j)| {
        let x = (-((i as f64 - (pos.0 as f64)).powi(2) / div)).exp();
        let y = (-((j as f64 - (pos.1 as f64)).powi(2) / div)).exp();
        x * y
    };

    Array2::from_shape_fn(dims, shape_fn)
}

/// Returns the [Euclidean distance] between `a` and `b`.
///
/// [Euclidean distance]: https://en.wikipedia.org/wiki/Euclidean_distance
fn euclid_dist(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    debug_assert_eq!(a.len(), b.len());

    a.iter()
        .zip(b.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Unit tests
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

        assert_eq!(map.winner(Array1::from(vec![1.5; 5])), (1, 1));
        assert_eq!(map.winner(Array1::from(vec![0.5; 5])), (0, 0));
    }

    #[test]
    fn test_euclid() {
        let a = array![1.0, 2.0, 3.0, 4.0];
        let b = array![4.0, 5.0, 6.0, 7.0];

        assert_eq!(euclid_dist(a.view(), b.view()), 6.0);
    }
}
