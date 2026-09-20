#pragma once

#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <iostream>

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
    mutable bool initialized_;
    Grid(std::size_t rows, std::size_t cols)
      : rows_{rows}
    , cols_{cols}
    , stride_{(cols_ + 7) & (~7)}
    , data{nullptr}
    , initialized_{false}
    {
      // check if stride is power of two to remove cache aliasing
      if ((stride_ & (stride_ - 1)) == 0,0) {
        stride_ += cols_ + 8;
      } 

      data = static_cast<double*>(std::malloc(rows * stride_ * sizeof(double))); 
    };


    double& operator()(std::size_t i, std::size_t j) {return data[i * stride_ + j];};
    double  operator()(std::size_t i, std::size_t j) const {return data[i * stride_ + j];};
    std::size_t rows() const{return rows_;};
    std::size_t cols() const{return cols_;};
    std::size_t stride() const{return stride_;};
    bool& initialized() const{return initialized_;};
    
    //////////////////////////////////////////////////////////////////////////////////////////
    // boilerplate for safety
    //////////////////////////////////////////////////////////////////////////////////////////

    ~Grid() {
      std::free (data);
    }

    // copy constructor
    Grid(const Grid&) = delete;
    // copy assignment
    Grid& operator=(const Grid&) = delete;

    // move constructor
    Grid(Grid&& other) noexcept
        : rows_{other.rows_}, cols_{other.cols_}, stride_{other.stride_}, data{other.data}, initialized_{other.initialized_} {
        other.data = nullptr;
        other.rows_ = 0;
        other.cols_ = 0;
      other.stride_ = 0;
      other.initialized_ = false;
    }

    //move assignment
    Grid& operator=(Grid&& other) noexcept {
        if (this != &other) {
            std::free(data);
            rows_ = other.rows_;
            cols_ = other.cols_;
            stride_ = other.stride_;
            initialized_ = other.initialized_;
            data = other.data;
            other.data = nullptr;
            other.rows_ = 0;
            other.cols_ = 0;
            other.stride_ = 0;
            other.initialized_ = false;
        }
        return *this;
    }
    //////////////////////////////////////////////////////////////////////////////////////////
    // end of boilerplate for safety
    //////////////////////////////////////////////////////////////////////////////////////////
    
};  

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid. Implement your solution here.
void apply_stencil(const Grid& old_grid, Grid& new_grid) {
  const std::size_t rows = old_grid.rows();
  const std::size_t cols = old_grid.cols();
  const std::size_t stride = old_grid.stride();

  // reduces copies
  if(!new_grid.initialized()) {
    old_grid.initialized() = true;
    new_grid.initialized() = true;
    std::memcpy(&new_grid.data[0], &old_grid.data[0], cols * sizeof(double));
    std::memcpy(
        new_grid.data + stride * (rows - 1), // we need only to copy until cols
        old_grid.data + stride * (rows - 1), // the remaining values aren't used
        cols * sizeof(double)
        );
    for(std::size_t i = 1; i < rows - 1; ++i) {
      (new_grid.data + i * stride)[0] = (old_grid.data + i * stride)[0];
      (new_grid.data + i * stride)[cols - 1] = (old_grid.data + i * stride)[cols - 1];
    }
  }

#pragma omp parallel for schedule(static)
  for(std::size_t i = 1; i < rows - 1; ++i) {
    const double* __restrict__ prev = old_grid.data + (i-1) * stride;
    const double* __restrict__ cur = old_grid.data + i * stride;
    const double* __restrict__ next = old_grid.data + (i+1) * stride;
    double* __restrict__ to = new_grid.data + i * stride;

    for(std::size_t j = 1; j < cols -1 ; ++j) {
      to[j] = std::fma(((prev[j] + next[j]) + (cur[j-1]+ cur[j+1])), 0.125, cur[j] * 0.5);
      // to[j] = ((prev[j] + next[j]) + (cur[j-1]+ cur[j+1]) + cur[j] * 4.0) * 0.125;
    }
  }
};


