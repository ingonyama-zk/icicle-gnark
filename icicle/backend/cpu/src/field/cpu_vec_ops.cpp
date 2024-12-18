
#include "icicle/backend/vec_ops_backend.h"
#include "icicle/errors.h"
#include "icicle/runtime.h"
#include "icicle/utils/log.h"

#include "icicle/fields/field_config.h"
#include "tasks_manager.h"
#include <cstdint>
#include <sys/types.h>
#include <vector>

using namespace field_config;
using namespace icicle;

/* Enumeration for the selected operation to execute.
 * The worker task is templated by this enum and based on that the functionality is selected. */
enum VecOperation {
  VECTOR_ADD,
  VECTOR_SUB,
  VECTOR_MUL,
  CONVERT_TO_MONTGOMERY,
  CONVERT_FROM_MONTGOMERY,
  SCALAR_MUL_VEC,

  NOF_VECTOR_OPERATIONS
};

/**
 * @class VectorOpTask
 * @brief Contains all the functionality that a single worker can execute for any vector operation.
 *
 * The enum VecOperation defines which functionality to execute.
 * Based on the enum value, the functionality is selected and the worker execute that function for every task that
 * dispatched by the manager.
 */
template <typename T>
class VectorOpTask : public TaskBase
{
public:
  // Constructor
  VectorOpTask() : TaskBase() {}

  // Set the operands to execute a task of 2 operands and 1 output and dispatch the task
  void send_2ops_task(
    VecOperation operation,
    const uint32_t nof_operations,
    const T* op_a,
    const T* op_b,
    const uint32_t stride,
    T* output)
  {
    m_operation = operation;
    m_nof_operations = nof_operations;
    m_op_a = op_a;
    m_op_b = op_b;
    m_stride = stride;
    m_output = output;
    dispatch();
  }

  // Set the operands to execute a task of 1 operand and 1 output and dispatch the task
  void send_1op_task(VecOperation operation, const uint32_t nof_operations, const T* op_a, T* output)
  {
    m_operation = operation;
    m_nof_operations = nof_operations;
    m_op_a = op_a;
    m_output = output;
    dispatch();
  }

  // Execute the selected function based on m_operation
  virtual void execute() { (this->*functionPtrs[static_cast<size_t>(m_operation)])(); }

private:
  // Single worker functionality to execute vector add (+)
  void vector_add()
  {
    for (uint64_t i = 0; i < m_nof_operations; ++i) {
      m_output[i] = m_op_a[i] + m_op_b[i];
    }
  }

  // Single worker functionality to execute vector add (+)
  void vector_sub()
  {
    for (uint64_t i = 0; i < m_nof_operations; ++i) {
      m_output[i] = m_op_a[i] - m_op_b[i];
    }
  }
  // Single worker functionality to execute vector mul (*)
  void vector_mul()
  {
    for (uint64_t i = 0; i < m_nof_operations; ++i) {
      m_output[i] = m_op_a[i] * m_op_b[i];
    }
  }
  // Single worker functionality to execute conversion from barret to montgomery
  void convert_to_montgomery()
  {
    for (uint64_t i = 0; i < m_nof_operations; ++i) {
      m_output[i] = T::to_montgomery(m_op_a[i]);
    }
  }
  // Single worker functionality to execute conversion from montgomery to barret
  void convert_from_montgomery()
  {
    for (uint64_t i = 0; i < m_nof_operations; ++i) {
      m_output[i] = T::from_montgomery(m_op_a[i]);
    }
  }
  // Single worker functionality to execute scalar * vector
  void scalar_mul_vec()
  {
    for (uint64_t i = 0; i < m_nof_operations; ++i) {
      m_output[m_stride * i] = *m_op_a * m_op_b[m_stride * i];
    }
  }

  // An array of available function pointers arranged according to the VecOperation enum
  using FunctionPtr = void (VectorOpTask::*)();
  static constexpr std::array<FunctionPtr, static_cast<int>(NOF_VECTOR_OPERATIONS)> functionPtrs = {
    &VectorOpTask::vector_add,              // VECTOR_ADD,
    &VectorOpTask::vector_sub,              // VECTOR_SUB,
    &VectorOpTask::vector_mul,              // VECTOR_MUL,
    &VectorOpTask::convert_to_montgomery,   // CONVERT_TO_MONTGOMERY,
    &VectorOpTask::convert_from_montgomery, // CONVERT_FROM_MONTGOMERY,
    &VectorOpTask::scalar_mul_vec,          // SCALAR_MUL_VEC,

  };

  VecOperation m_operation;  // the operation to execute
  uint32_t m_nof_operations; // number of operations to execute for this task
  const T* m_op_a;           // pointer to operand A. Operand A is a vector, or matrix in case of replace_elements
  const T* m_op_b;           // pointer to operand B. Operand B is a vector or scalar
  uint64_t m_start_index;    // index used in bitreverse operation and out of place matrix transpose
  uint64_t m_stop_index;     // index used in reduce operations and out of place matrix transpose
  uint32_t m_bit_size;       // use in bitrev operation
  uint64_t m_stride;         // used to support column batch operations
  uint64_t m_stride_out;     // used in slice operation
  T*
    m_output; // pointer to the output. Can be a vector, scalar pointer, or a matrix pointer in case of replace_elements
  uint32_t m_log_nof_rows; // log of the number of rows in the matrix, used in replace_elements
  uint32_t m_log_nof_cols; // log of the number of columns in the matrix, used in replace_elements
  uint32_t m_nof_rows;     // the number of rows in the matrix, used in out of place matrix transpose
  uint32_t m_nof_cols;     // the number of columns in the matrix, used in out of place matrix transpose
  const std::vector<uint64_t>* m_start_indices_in_mat; // Indices used in replace_elements operations

public:
  T m_intermidiate_res;    // pointer to the output. Can be a vector or scalar pointer
  uint64_t m_idx_in_batch; // index in the batch. Used in intermediate res tasks
};                         // class VectorOpTask

#define NOF_OPERATIONS_PER_TASK 512
#define CONFIG_NOF_THREADS_KEY  "n_threads"

// extract the number of threads to run from config
int get_nof_workers(const VecOpsConfig& config)
{
  if (config.ext && config.ext->has(CONFIG_NOF_THREADS_KEY)) { return config.ext->get<int>(CONFIG_NOF_THREADS_KEY); }

  int hw_threads = std::thread::hardware_concurrency();
  return ((hw_threads > 1) ? hw_threads - 1 : 1); // reduce 1 for the main
}

// Execute a full task from the type vector = vector (op) vector
template <typename T>
eIcicleError
cpu_2vectors_op(VecOperation op, const T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output)
{
  TasksManager<VectorOpTask<T>> task_manager(get_nof_workers(config) - 1);
  const uint64_t total_nof_operations = size * config.batch_size;
  for (uint64_t i = 0; i < total_nof_operations; i += NOF_OPERATIONS_PER_TASK) {
    VectorOpTask<T>* task_p = task_manager.get_idle_or_completed_task();
    task_p->send_2ops_task(
      op, std::min((uint64_t)NOF_OPERATIONS_PER_TASK, total_nof_operations - i), vec_a + i, vec_b + i, 1, output + i);
  }
  task_manager.wait_done();
  return eIcicleError::SUCCESS;
}

// Execute a full task from the type vector = scalar (op) vector
template <typename T>
eIcicleError cpu_scalar_vector_op(
  VecOperation op, const T* scalar_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output)
{
  TasksManager<VectorOpTask<T>> task_manager(get_nof_workers(config) - 1);
  const uint64_t total_nof_operations = size;
  const uint32_t stride = config.columns_batch ? config.batch_size : 1;
  for (uint32_t idx_in_batch = 0; idx_in_batch < config.batch_size; idx_in_batch++) {
    for (uint64_t i = 0; i < total_nof_operations; i += NOF_OPERATIONS_PER_TASK) {
      VectorOpTask<T>* task_p = task_manager.get_idle_or_completed_task();
      task_p->send_2ops_task(
        op, std::min((uint64_t)NOF_OPERATIONS_PER_TASK, total_nof_operations - i), scalar_a + idx_in_batch,
        config.columns_batch ? vec_b + idx_in_batch + i * config.batch_size : vec_b + idx_in_batch * size + i, stride,
        config.columns_batch ? output + idx_in_batch + i * config.batch_size : output + idx_in_batch * size + i);
    }
  }
  task_manager.wait_done();
  return eIcicleError::SUCCESS;
}

///////////////////////////////////////////////////////
// Functions to register at the CPU backend
/*********************************** ADD ***********************************/
template <typename T>
eIcicleError cpu_vector_add(
  const Device& device, const T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output)
{
  return cpu_2vectors_op(VecOperation::VECTOR_ADD, vec_a, vec_b, size, config, output);
}

REGISTER_VECTOR_ADD_BACKEND("CPU", cpu_vector_add<scalar_t>);

/*********************************** SUB ***********************************/
template <typename T>
eIcicleError cpu_vector_sub(
  const Device& device, const T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output)
{
  return cpu_2vectors_op(VecOperation::VECTOR_SUB, vec_a, vec_b, size, config, output);
}

REGISTER_VECTOR_SUB_BACKEND("CPU", cpu_vector_sub<scalar_t>);

/*********************************** MUL ***********************************/
template <typename T>
eIcicleError cpu_vector_mul(
  const Device& device, const T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output)
{
  return cpu_2vectors_op(VecOperation::VECTOR_MUL, vec_a, vec_b, size, config, output);
}

REGISTER_VECTOR_MUL_BACKEND("CPU", cpu_vector_mul<scalar_t>);

/*********************************** CONVERT MONTGOMERY ***********************************/
template <typename T>
eIcicleError cpu_convert_montgomery(
  const Device& device, const T* input, uint64_t size, bool is_to_montgomery, const VecOpsConfig& config, T* output)
{
  TasksManager<VectorOpTask<T>> task_manager(get_nof_workers(config) - 1);
  const uint64_t total_nof_operations = size * config.batch_size;
  for (uint64_t i = 0; i < total_nof_operations; i += NOF_OPERATIONS_PER_TASK) {
    VectorOpTask<T>* task_p = task_manager.get_idle_or_completed_task();
    task_p->send_1op_task(
      (is_to_montgomery ? CONVERT_TO_MONTGOMERY : CONVERT_FROM_MONTGOMERY),
      std::min((uint64_t)NOF_OPERATIONS_PER_TASK, total_nof_operations - i), input + i, output + i);
  }
  task_manager.wait_done();
  for (uint64_t i = 0; i < size * config.batch_size; i++) {}
  return eIcicleError::SUCCESS;
}

REGISTER_CONVERT_MONTGOMERY_BACKEND("CPU", cpu_convert_montgomery<scalar_t>);

/*********************************** MUL BY SCALAR***********************************/
template <typename T>
eIcicleError cpu_scalar_mul(
  const Device& device, const T* scalar_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output)
{
  return cpu_scalar_vector_op(VecOperation::SCALAR_MUL_VEC, scalar_a, vec_b, size, config, output);
}

REGISTER_SCALAR_MUL_VEC_BACKEND("CPU", cpu_scalar_mul<scalar_t>);
