--\! gpssh -f all_hosts -e 'rm -rf /tmp/user_result_*'
--\! gpssh -f all_hosts -e 'rm -rf /tmp/read_data*.log'
--\! gpssh -f all_hosts -e 'rm -rf /tmp/run_user_function_*.log'
\! gpscp -f segment_hosts /home/gpadmin/read_data.py =:/home/gpadmin
\! gpscp -f segment_hosts /home/gpadmin/keras_train.py =:/home/gpadmin
DROP external web table ext;
-- we need to make sure that this table is distributed in the same way as the source table in order
-- to prevent data reshuffling between the segments
--CREATE WRITABLE EXTERNAL WEB TABLE ext (x double precision[], y int, x_type text, y_type text, g int) EXECUTE 'python /home/gpadmin/read_data.py keras_train' FORMAT 'TEXT' (DELIMITER '|') DISTRIBUTED BY (g);
CREATE WRITABLE EXTERNAL WEB TABLE ext (x double precision[], y int, x_type text, y_type text, g int) EXECUTE 'python /home/gpadmin/read_data.py keras_train' FORMAT 'TEXT' (DELIMITER '|');
insert into ext select x, y, pg_typeof(x), pg_typeof(y) from cifar_train;



--CREATE READABLE EXTERNAL TABLE result_ext (seg_id int, model text)
--LOCATION ('file://mdw/tmp/local_model', 'file://sdw1/tmp/local_model' ) FORMAT 'TEXT' (DELIMITER '|');

-- create table model_output_table as select * from result_ext; -- ( this step can be optional)
-- drop external table if exists result_ext;
