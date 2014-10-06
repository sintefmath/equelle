/* Libraries */
var _ = require('underscore'),
    fs = require('fs-extra'),
    path = require('path'),
    mime = require('mime');
/* Own modules */
var config = require('./config.js'),
    helpers = require('./helpers.js');


(function(module) {
    /* Return an object containing the list of all examples, and some of their options */
    module.getList = function(res) {
        var quit = function(error) { res.writeHead(500); res.end() };
        var tryAsync = helpers.tryAsync('Examples server', quit);

        var readExamplesDir, findConfigFiles, checkExampleConfigs, readImageFiles, returnExamples;

        var exampleConfigs = {};
        var examples = [];

        /* Read the contents of the examples directory */
        readExamplesDir = function() { 
            tryAsync(true, fs.readdir, config.examples_dir)
            .complete(findConfigFiles)
            .run();
        };

        /* Loop through the subdirectories of the examples-directory, and look for config.json files */
        findConfigFiles = function(files) {
            var done = _.after(files.length, checkExampleConfigs);

            _.each(files, function(dir) {
                var path = config.examples_dir+'/'+dir;
                // Try to read the config.json file
                tryAsync(fs.readFile, path+'/config.json', { encoding: 'utf8' })
                .complete(function(data) {
                    // If it was a successfull read, add it to the list
                    exampleConfigs[path] = data;
                })
                .always(done)
                .run();
            });
        };

        /* Check the found config.json files for valid examples */
        checkExampleConfigs = function() {
            var done = _.after(_.keys(exampleConfigs).length, readImageFiles);

            _.each(exampleConfigs, function(data, dir) {
                // Try to parse the JSON config file
                try {
                    var config = JSON.parse(data);
                    var example = {
                         id: path.basename(dir)
                        ,name: config.name
                    };
                    // Check that all indicated input-files are present in the directory
                    tryAsync(fs.readdir, dir)
                    .complete(function(files) {
                        if (!_.contains(files, config.source) || (config.inputfiles && (_.difference(_.map(config.inputfiles, function(i) { return i.file }), files).length > 0))) {
                            helpers.logError('Examples server', 'Not all input files present in: '+dir);
                        } else {
                            // We have all neccesary files
                            if (config.description) example.description = config.description;
                            if (config.image) example.image = dir+'/'+config.image;
                            examples.push(example);
                        }
                    })
                    .always(done)
                    .run();
                } catch (err) {
                    helpers.logError('Examples server', err);
                }
            });
        };

        /* Read all example thumbnail, and attach data */
        readImageFiles = function() {
            var done = _.after(examples.length, returnExamples);

            _.each(examples, function(example) {
                if (example.image) {
                    tryAsync(fs.readFile, example.image)
                    .complete(function(data) {
                        // Create a data-url of the image data
                        var dataPrefix = 'data:'+mime.lookup(example.image)+';base64,';
                        var image = data.toString('base64');
                        example.image = dataPrefix+image;
                    })
                    .error(function() {
                        delete example.image;
                    })
                    .always(done)
                    .run();
                } else {
                    done();
                }
            });
        };

        /* Send response to client */
        returnExamples = function() {
            // Return everything we found
            var ret = { examples: examples };
            res.writeHead(200, { 'Content-Type': 'text/javascript' });
            res.write('var equelleKitchenSinkExamples = JSON.parse(\'');
            res.write(JSON.stringify(ret));
            res.end('\');');
        };

        // Run the function
        readExamplesDir();
    };

    /* Return an object containing the example data */
    module.getExample = function(id, res) {
        var quit = function(error) { res.writeHead(500); res.end() };
        var tryAsync = helpers.tryAsync('Examples server', quit);

        var readConfigFile, readSource, returnExample;

        var path = config.examples_dir+'/'+id;
        var exampleReturn = {};

        /* Read the config.json file */
        readConfigFile = function() {
            tryAsync(true, fs.readFile, path+'/config.json', { encoding: 'utf8' })
            .complete(function(data) {
                try {
                    var exConfig = JSON.parse(data);
                    // Add the grid settings to returned object
                    exampleReturn.grid = exConfig.grid;
                    // Add the input-files
                    exampleReturn.inputfiles = _.map(exConfig.inputfiles, function(file) {
                        return {
                            tag: file.tag,
                            name: file.file,
                            compressed: file.compressed
                        }
                    });
                    // Add source code
                    readSource(path+'/'+exConfig.source, exConfig.inputfiles);
                } catch (error) {
                    helpers.logError('Examples server', error);
                }
            })
            .run();
        };

        /* Read the Equelle source */
        readSource = function(path, inputfiles) {
            tryAsync(true, fs.readFile, path, { encoding: 'utf8' })
            .complete(function(data) {
                exampleReturn.source = data;

                returnExample();
            })
            .run();
        };

        /* Return the retrieved data */
        returnExample = function() {
            res.writeHead(200);
            res.end(JSON.stringify(exampleReturn));
        };

        // Run function
        readConfigFile();
    };

    /* Return the raw input-file data */
    module.getInputFile = function(id, file, res) {
        var quit = function(error) { res.writeHead(500); res.end() };
        var tryAsync = helpers.tryAsync('Examples server', quit);

        var readConfigFile, readFile;

        var path = config.examples_dir+'/'+id;
        /* Read the config.json file to make sure that we can share this file */
        readConfigFile = function() {
            tryAsync(true, fs.readFile, path+'/config.json', { encoding: 'utf8' })
            .complete(function(data) {
                try {
                    var exConfig = JSON.parse(data);

                    // Check that the inputfiles list contains this file
                    if (_.findWhere(exConfig.inputfiles, { file: file })) {
                        readFile(path+'/'+file);
                    } else {
                        throw 'Did not find file: "'+file+'"';
                    }
                } catch (error) {
                    helpers.logError('Examples server', error);
                    quit(error);
                }
            })
            .run();
        };

        /* Read the actual file data, and send to client */
        readFile = function(path) {
            tryAsync(true, fs.readFile, path)
            .complete(function(data) {
                // Write data to response
                res.writeHead(200, {
                    'Content-Type': 'application/octet-stream',
                    'Content-Length': data.length
                });
                res.end(data);
            })
            .run();
        };

        // Run function
        readConfigFile();
    };
})(module.exports);
