(function(){
    angular.module('eqksServer')
    /* A helper that attaches a socket connection to an event-emitting object, for use when communicating with the back-end */
    .factory('eqksPersistentSocket', [ function() { return {
        attach: function(object, url, protocol, keepAlive) {
            /* Store the sockets state here */
            var state = {
                lastStatus: 'closed',
                expecting: [],
                connectWait: 100,
                done: false
            };
            // Default wait is 30 seconds
            var timeoutWait = keepAlive || 30*1000;
            var killPromise;
            var resetKillTimer = function() {
                clearTimeout(killPromise);
                setTimeout(function() {
                    if (state.lastStatus != 'ready') {
                        // Don't close the socket if it is doing something
                        resetKillTimer();
                    } else {
                        state.timeout = true;
                        if (state.socket && !(state.socket.readyState === WebSocket.CLOSING || state.socket.readyState === WebSocket.CLOSED)) {
                            console.log('Killing socket');
                            state.socket.close();
                        }
                    }
                }, timeoutWait);
            };

            /* Attach methods for outside use */
            var connectPromise;
            object._reset = function() {
                // Close connection if one is open
                if (state.socket && (state.socket.readyState === WebSocket.CONNECTING || state.socket.readyState === WebSocket.OPEN)) {
                    state.socket.close();
                }

                // Wait for a little time in case something has happened to the server
                clearTimeout(connectPromise);
                setTimeout(function() {
                    // Open a new connection
                    object._connect();
                }, state.connectWait);
                state.connectWait *= 2;
            };

            object._connect = function() {
                clearTimeout(connectPromise);

                state.done = false;

                // Check if we already have a socket
                if (state.socket && state.socket.readyState === WebSocket.CONNECTING) return;
                if (state.socket && state.socket.readyState === WebSocket.OPEN && state.lastStatus == 'ready') {
                    setTimeout(function() {
                        object.trigger('socketReady');
                    }, 0);
                    return;
                }

                // If we are removing an old socket, remove the event handlers
                if (state.socket) {
                    state.socket.onerror = null;
                    state.socket.onclose = null;
                    state.socket.onmessage = null;
                }

                // Remove old expecters
                state.expecting = [];

                // Clear old timeout-kill
                state.timeout = false;

                // Create a new socket connection to the server
                console.log('Connecting to server...');
                state.socket = new WebSocket(url, protocol);

                // Bind to socket events
                state.socket.onopen = function(event) {
                    // Reset the exponentially growing connect-wait
                    state.connectWait = 100;
                };

                state.socket.onerror = function(event) {
                    // Pass event on
                    object.trigger('socketError', event);
                    // Reset the connection
                    state.lastStatus = 'error';
                    object._reset();
                };

                state.socket.onclose = function(event) {
                    console.log('Socket closing, state:', state);
                    state.lastStatus = 'closed';
                    if (!state.timeout && this == state.socket) {
                        if (!state.done) {
                            // If it was not closed deliberately, emit error
                            object.trigger('socketError', 'Unexpected closure of socket');
                            state.lastStatus = 'error';
                        }
                        // And re-open it
                        object._reset();
                    }
                };

                state.socket.onmessage = function(event) {
                    resetKillTimer();

                    if (event.data instanceof ArrayBuffer || event.data instanceof Blob) {
                        object.trigger('socketData', event.data);
                    } else {
                        try {
                            var msg = JSON.parse(event.data);
                            // Parse message we got from server
                            if (!msg.status) throw('No status in message');
                            state.lastStatus = msg.status;

                            // Special ready status
                            if (msg.status == 'ready') {
                                resetKillTimer();
                                object.trigger('socketReady');
                                return;
                            }
                            // Special failed status
                            if (msg.status == 'error' || msg.status == 'failed') {
                                throw msg.error;
                            }

                            // Check if anyone is expecting this status
                            var expecter = _.findWhere(state.expecting, { status: msg.status });
                            if (expecter) {
                                state.expecting = _.without(state.expecting, expecter);
                                expecter.cb(msg);
                            } else {
                                object.trigger('socketMessage', msg);
                            }
                        } catch (e) {
                            object.trigger('socketError', e);
                            state.lastStatus = 'error';
                            object._reset();
                        }
                    }
                };

                resetKillTimer();
            };

            object._send = function(obj) {
                resetKillTimer();

                if (state.socket && state.socket.readyState === WebSocket.OPEN) {
                    if (typeof obj === 'string' || obj instanceof ArrayBuffer || obj instanceof Blob) {
                        // Send as is
                        state.socket.send(obj);
                    } else {
                        // Serialize as JSON
                        state.socket.send(JSON.stringify(obj));
                    }
                } else {
                    setTimeout(function() {
                        object.trigger('socketError', new ErrorEvent('Socket not ready'));
                    }, 0);
                }
            };

            object._expect = function(status, cb) {
                state.expecting.push({
                    status: status,
                    cb: cb
                });
            };

            object._sendExpect = function(obj, status, cb) {
                object._send(obj);
                object._expect(status, cb);
            };

            object._done = function() {
                state.done = true;
                state.lastStatus = 'done';
                console.log('SOCKET marked as done');
            };
        }
    }}])
})();
