
mosek_datapath = '/scail/u/vigneshr/joint_nlp_vision/codes/3rd_party/mosek/';

% setup mosek
mosek_license = [mosek_datapath 'license/mosek.lic'];
mosek_path = [mosek_datapath '7/toolbox/r2012b/'];
addpath(mosek_path);
setenv('MOSEKLM_LICENSE_FILE', mosek_license);
