# RustySOM
Rust library for Self Organising Maps (SOM).

## API

Use `SOM::create` to create an SOM object using the API call below, which creates an SOM with `length x breadth` cells and accepts neurons of length `inputs`.

```rust
pub fn create(length: usize, breadth: usize, inputs: usize, randomize: bool, learning_rate: Option<f32>, sigma: Option<f32>, decay_function: Option<fn(f32, u32, u32) -> f64>, neighbourhood_function: Option<fn((usize, usize), (usize, usize), f32) -> Array2<f64>>) -> SOM { ... }
```

`randomize` is a flag, which, if true, initializes the weights of each cell to random, small, floating-point values.

`learning_rate`, optional, is the learning_rate of the SOM; by default it will be `0.5`.

`sigma`, optional, is the spread of the neighbourhood function; by default it will be `1.0`.

`decay_function`, optional, is a function pointer that accepts functions that take 3 parameters of types `f32, u32, u32`, and returns an `f64`. This function is used to "decay" both the `learning_rate` and `sigma`. By default it is

    new_value = old_value / (1 + current_iteration/total_iterations)

`neighbourhood_function`, optional, is also a function pointer that accepts functions that take 3 parameters, a tuple of type `(usize, usize)` representing the size of the SOM, another tuple of type `(usize, usize)` representing the position of the winner neuron, and an `f32` representing `sigma`; and returns a 2D Array containing weights of the neighbours of the winning neuron, i.e, centered at `winner`. By default, the Gaussian function will be used, which returns a "Gaussian centered at the winner neuron".

---

Use `SOM_Object.train_random()` to train the SOM with the input dataset, where samples from the input dataset are picked in a random order.

```rust
pub fn train_random(&mut self, data: Array2<f64>, iterations: u32) { ... }
```

Samples (rows) from the 2D Array `data` are picked randomly and the SOM is trained for `iterations` iterations!

---

Use `SOM_Object.train_batch()` to train the SOM with the input dataset, where samples from the input dataset are picked in a sequential order.

```rust
pub fn train_batch(&mut self, data: Array2<f64>, iterations: u32) { ... }
```

Samples (rows) from the 2D Array `data` are picked sequentially and the SOM is trained for `iterations` iterations!

---

Use `SOM_Object.winner()` to find the winning neuron for a given sample.

```rust
pub fn winner(&mut self, elem: Array1<f64>) -> (usize, usize) { ... }
```

This function must be called **with** an SOM object. 

Requires one parameter, a 1D Array of `f64`s representing the input sample. 

Returns a tuple `(usize, usize)` representing the x and y coordinates of the winning neuron in the SOM.

---

Use `SOM_Object.winner_dist()` to find the winning neuron for a given sample, and it's distance from this winner neuron.

```rust
pub fn winner_dist(&mut self, elem: Array1<f64>) -> ((usize, usize), f64) { ... }
```

This function must be called **with** an SOM object. 

Requires one parameter, a 1D Array of `f64`s representing the input sample. 

Returns a tuple `(usize, usize)` representing the x and y coordinates of the winning neuron in the SOM.

Also returns an `f64` representing the distance of the input sample from this winner neuron.

---

```rust
pub fn activation_response(&self) -> ArrayView2<usize> { ... }
```

This function returns the activation map of the SOM. The activation map is a 2D Array where each cell at `(i, j)` represents the number of times the `(i, j)` cell of the SOM was picked to be the winner neuron.

---

```rust
pub fn get_size(&self) -> (usize, usize)
```

This function returns a tuple representing the size of the SOM. Format is `(length, breadth)`.

---

## Primary Contributors

|   |   |
|:-:|:-:|
| <img src="https://github.com/aditisrinivas97.png" width="75"> | [Aditi Srinivas](https://github.com/aditisrinivas97) |
| <img src="https://github.com/avinashshenoy97.png" width="75"> | [Avinash Shenoy](https://github.com/avinashshenoy97) |