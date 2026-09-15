#pragma once

#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <cmath>

// Starter Grid for the 2D heat-diffusion problem.
//
// The evaluation harness uses operator() to set initial conditions and to read
// results; it never touches your internal storage. Keep this interface,
// everything else is yours.
class Grid {
  private:
    std::size_t rows_;
    std::size_t cols_;
    std::size_t stride_;
    // additions from here

  public:
    double* data;
    Grid(std::size_t rows, std::size_t cols)
      : rows_{rows}
    , cols_{cols}
    , stride_{(cols + 8)} // I have no idea why but this improves throughput
    , data{nullptr}       // might be something to do with cache aliasing
    {
      // data = static_cast<double*>(std::aligned_alloc(64, rows * stride_ * sizeof(double)));
      data = static_cast<double*>(std::malloc(rows * stride_ * sizeof(double))); // aligned alloc doens't change anything it seems
                                                                                 // iterating from 1 to  is inherently unaligned throughput
                                                                                 // might want to experiment more with it

      #pragma omp parallel for schedule(static)
      for(size_t i = 0; i < rows * stride_; ++i){
        data[i] = 0.0;
      }

    };
    ~Grid() {
      std::free(data);
    }

    double& operator()(std::size_t i, std::size_t j) {return data[i * stride_ + j];};
    double  operator()(std::size_t i, std::size_t j) const {return data[i * stride_ + j];};
    std::size_t rows() const{return rows_;};
    std::size_t cols() const{return cols_;};
    std::size_t stride() const{return stride_;};
    
    // copy constructor
    Grid(const Grid&) = delete;
    // copy assignment
    Grid& operator=(const Grid&) = delete;

    // move constructor
    Grid(Grid&& other) noexcept
        : rows_{other.rows_}, cols_{other.cols_}, stride_{other.stride_}, data{other.data} {
        other.data = nullptr;
        other.rows_ = 0;
        other.cols_ = 0;
        other.stride_ = 0;
    }

    //move assignment
    Grid& operator=(Grid&& other) noexcept {
        if (this != &other) {
            std::free(data);
            rows_ = other.rows_;
            cols_ = other.cols_;
            stride_ = other.stride_;
            data = other.data;
            other.data = nullptr;
            other.rows_ = other.cols_ = other.stride_ = 0;
        }
        return *this;
    }
};  

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid. Implement your solution here.
void apply_stencil(const Grid& old_grid, Grid& new_grid) {
  const std::size_t rows = old_grid.rows();
  const std::size_t cols = old_grid.cols();
  const std::size_t stride = old_grid.stride();

  std::memcpy(&new_grid.data[0], &old_grid.data[0], cols * sizeof(double));
  std::memcpy(
      new_grid.data + stride * (rows - 1), // we need only to copy until cols
      old_grid.data + stride * (rows - 1), // the remaining values aren't used
      cols * sizeof(double)
      );

// #pragma omp parallel for schedule(static)
  for(std::size_t i = 1; i < rows - 1; ++i) {
    const double* __restrict__ prev = old_grid.data + (i-1) * stride;
    const double* __restrict__ cur = old_grid.data + i * stride;
    const double* __restrict__ next = old_grid.data + (i+1) * stride;
    double* __restrict__ to = new_grid.data + i * stride;
    to[0] = cur[0];
    to[cols - 1] = cur[cols - 1];

// compiler already vectorizes this without flag
    for(std::size_t j = 1; j < cols - 1; ++j) {
      to[j] = std::fma(((prev[j] + next[j]) + (cur[j-1]+ cur[j+1])), 0.125, cur[j] * 0.5);
    }
  }

};


