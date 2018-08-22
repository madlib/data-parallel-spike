\! gpssh -f all_hosts -e 'rm -rf /tmp/local_model'
\! gpssh -f all_hosts -e 'rm -rf /tmp/read_data.log'
\! gpssh -f all_hosts -e 'rm -rf /tmp/keras_train.log'
\! gpscp -f segment_hosts /home/gpadmin/read_data.py =:/home/gpadmin
\! gpscp -f segment_hosts /home/gpadmin/keras_train.py =:/home/gpadmin
DROP external web table mnist_web_ext;
CREATE WRITABLE EXTERNAL WEB TABLE mnist_web_ext (x double precision[], y int, x_type text, y_type text) EXECUTE 'python /home/gpadmin/keras_train.py' FORMAT 'TEXT' (DELIMITER '|');
insert into mnist_web_ext select x, y, pg_typeof(x), pg_typeof(y) from mnist_train;