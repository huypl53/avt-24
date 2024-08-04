SELECT * FROM public.avt_task where task_type = 5

-- update public.avt_task set task_stat = 69 where task_type = 5 and id = 68


-- check haning query
-- select *  from pg_stat_activity


-- to kill query
-- SELECT pg_cancel_backend(46398)