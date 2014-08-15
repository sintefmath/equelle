/* Return the configuration object */
module.exports.equelle_dir = '/equelle/build';
module.exports.equelle_compiler = module.exports.equelle_dir+'/compiler/ec';
module.exports.compiler_skel_dir = '/scripts/cppcompilerskel';
module.exports.examples_dir = '/srv/examples';

/* Read secret key from file */
module.exports.secret_key = require('fs').readFileSync('/srv/server/secretkey', { encoding: 'utf8' });
