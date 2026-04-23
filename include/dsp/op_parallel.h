#pragma once

typedef void (*op_parallel_for_rows_fn_t)(void *ctx, int row_begin, int row_end);

// Execute row ranges in parallel on default worker pool.
// Returns 0 on success and negative on bad arguments.
int op_parallel_for_rows(int n_rows, int min_rows_per_task, op_parallel_for_rows_fn_t fn, void *ctx);

