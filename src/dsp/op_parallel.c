#include "dsp/op_parallel.h"

#ifndef HTP_OPS_SIM_TEST
#include "dsp/worker_pool.h"
#endif

typedef struct {
#ifndef HTP_OPS_SIM_TEST
  EXPAND_COMMON_TASK_STATE_MEMBERS
#endif
  int                      n_rows;
  int                      rows_per_task;
  op_parallel_for_rows_fn_t fn;
  void                    *ctx;
} op_parallel_rows_task_state_t;

#ifndef HTP_OPS_SIM_TEST
static void op_parallel_for_rows_worker_loop(void *data, int _worker_index) {
  (void) _worker_index;
  op_parallel_rows_task_state_t *state = (op_parallel_rows_task_state_t *) data;

  while (1) {
    unsigned int task_id = worker_pool_atomic_inc_return(&(state->task_id)) - 1;
    if (task_id >= (unsigned int) state->n_tasks) {
      break;
    }

    int row_begin = (int) task_id * state->rows_per_task;
    int row_end   = row_begin + state->rows_per_task;
    if (row_end > state->n_rows) {
      row_end = state->n_rows;
    }
    state->fn(state->ctx, row_begin, row_end);
  }

  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}
#endif

int op_parallel_for_rows(int n_rows, int min_rows_per_task, op_parallel_for_rows_fn_t fn, void *ctx) {
  if (n_rows <= 0 || min_rows_per_task <= 0 || !fn) {
    return -1;
  }

#ifdef HTP_OPS_SIM_TEST
  // SIM backend: deterministic chunk scheduling without worker_pool dependency.
  for (int row_begin = 0; row_begin < n_rows; row_begin += min_rows_per_task) {
    int row_end = row_begin + min_rows_per_task;
    if (row_end > n_rows) {
      row_end = n_rows;
    }
    fn(ctx, row_begin, row_end);
  }
  return 0;
#else
  int n_workers = (int) num_hvx128_contexts;
  if (n_workers <= 1 || n_rows <= min_rows_per_task) {
    fn(ctx, 0, n_rows);
    return 0;
  }
  if (n_workers > n_rows) {
    n_workers = n_rows;
  }

  int n_chunks_per_task = min_rows_per_task;

  op_parallel_rows_task_state_t state;
  INIT_COMMON_TASK_STATE_MEMBERS(state, n_rows, n_chunks_per_task);
  state.n_rows        = n_rows;
  state.rows_per_task = n_chunks_per_task;
  state.fn            = fn;
  state.ctx           = ctx;

  worker_pool_job_t job = {
    .fptr = op_parallel_for_rows_worker_loop,
    .dptr = &state,
  };

  worker_pool_synctoken_init(&(state.sync_ctx), n_workers);
  for (int i = 0; i < n_workers; ++i) {
    worker_pool_submit(NULL, job);
  }
  worker_pool_synctoken_wait(&(state.sync_ctx));
  return 0;
#endif
}
