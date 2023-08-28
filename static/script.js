jQuery(function($) {
    // ONLOAD
    $(document).ready(function() {
        // Call the function to add nodes
        addNodes().then(() => {
            addWeights();
        });
    });
});

function addNodes() {
    return new Promise((resolve) => {
        fetch('/api/data')
            .then(response => response.json())
            .then(data => {
                console.log(data);
                var sizes = data.layer_sizes;

                // ######## NODES ########

                // Function to update nodes based on min and max inputs
                function updateNodes(layerDiv) {
                    let layerIndex = $('main').children().index(layerDiv);

                    var minInput = $('#bottom_header').children().eq(layerIndex).find('.min_input');
                    var maxInput = $('#bottom_header').children().eq(layerIndex).find('.max_input');

                    // Convert inputs to numbers and handle invalid inputs
                    var minValue = parseFloat(minInput.val()) || 1;
                    var maxValue = parseFloat(maxInput.val()) || sizes[layerIndex];

                    // Check if in range
                    if (minValue < 1) {
                        minInput.val(1);
                        minValue = 1;
                    }
                    if (maxValue > sizes[layerIndex]) {
                        maxInput.val(sizes[layerIndex]);
                        maxValue = sizes[layerIndex];
                    }

                    // Clear previous nodes
                    layerDiv.find('.node').remove();

                    // Create a <div> for each node in the layer
                    for (var i = minValue-1; i < maxValue; i++) {
                        var nodeDiv = $('<div class="node" id="node-'+layerIndex+'-'+i+'"></div>');
                        nodeDiv.text(i+1);
                        layerDiv.append(nodeDiv);
                    }
                }

                // Event listener for input changes
                $('.min_input, .max_input').on('input', function() {
                    var controlDiv = $(this).closest('.layer_controls');
                    var layerIndex = $('#bottom_header').children().index(controlDiv);
                    var layerDiv = $('main').children().eq(layerIndex)
                    updateNodes(layerDiv);
                });

                // Set initial min and max values
                $('.min_input').each(function() {
                    $(this).val(1);
                    $(this).attr('min', 1);
                });
                $('.max_input').each(function(index) {
                    $(this).val(10);//sizes[index]);
                    $(this).attr('max', sizes[index]);
                });

                // Update nodes on initial load
                $('.layer_column').each(function() {
                    var layerDiv = $(this);
                    updateNodes(layerDiv);
                });

                resolve();
            });
    });
}

function addWeights() {
    // Create a PixiJS application
    const app = new PIXI.Application({
        width: parseInt($('body').css('width')),
        height: parseInt($('body').css('height')),
        antialias: true,
        backgroundColor: 0x222222, // Set the background color
    });

    // Append the PIXI canvas to the container element
    var container = document.getElementById('PixiBox');
    container.appendChild(app.view);

    // Create a PixiJS graphics object
    var graphics = new PIXI.Graphics();
    app.stage.addChild(graphics);

    function updateWeights() {
        graphics.clear(); // Clear the graphics object
        console.log("dimentions", $('body').css('width'), $('body').css('height'));
        $('PixiBox').css('width', $('body').css('width'));
        $('PixiBox').css('height', $('body').css('height'));
        app.renderer.resize(parseInt($('body').css('width')), parseInt($('body').css('height')));

        layer_ranges = [[$('#input_column_controls .min_input').val(), $('#input_column_controls .max_input').val()], [$('#hidden1_column_controls .min_input').val(), $('#hidden1_column_controls .max_input').val()], [$('#hidden2_column_controls .min_input').val(), $('#hidden2_column_controls .max_input').val()], [$('#output_column_controls .min_input').val(), $('#output_column_controls .max_input').val()]]
        //console.log(layer_ranges);

        // Fetch data and draw lines
        fetch('/api/data')
            .then(response => response.json())
            .then(data => {
                console.log(data);
                // Find the minimum and maximum weight values
                let minWeight = Math.min(...data.weights.flat().flat().filter(weight => !isNaN(weight)));
                let maxWeight = Math.max(...data.weights.flat().flat().filter(weight => !isNaN(weight)));
                let maxAbsWeight = Math.max(Math.abs(minWeight), Math.abs(maxWeight));
                //console.log("Maxes:", minWeight, maxWeight, maxAbsWeight);

                // Function to draw lines between nodes
                function drawLine(startX, startY, endX, endY, weight) {
                    let lineColor = 'transparent'; // Default color is transparent

                    if (weight !== 0) {
                        const opacity = Math.abs(weight) / maxAbsWeight; // Calculate opacity based on weight
                        //console.log("opacity", opacity,  Math.abs(weight), maxWeight);
                        lineColor = weight > 0 ? `rgba(0, 100, 255, ${opacity})` : `rgba(255, 30, 0, ${opacity})`; // Set color based on weight sign
                    }

                    graphics.lineStyle(2, lineColor); // Set line thickness and color
                    graphics.moveTo(startX, startY); // Move to starting point
                    graphics.lineTo(endX, endY); // Draw line to ending point
                }

                window.requestAnimationFrame(() => {
                    //console.log(data.weights.length);
                    // ######## WEIGHTS ########
                    for (var i = 0; i < data.weights.length; i++) {
                        for (var j = layer_ranges[i][0]-1; j < layer_ranges[i][1]; j++) {
                            for (var k = layer_ranges[i+1][0]-1; k < layer_ranges[i+1][1]; k++) {
                                var node1 = $('#node-' + i + '-' + j);
                                var node2 = $('#node-' + (i + 1) + '-' + k);

                                // Get positions of nodes
                                var node1Offset = node1.offset();
                                var node2Offset = node2.offset();

                                // Get coordinates of node centers relative to their parent containers
                                var node1X = node1Offset.left + node1.width() / 2;
                                var node1Y = node1Offset.top + node1.height() / 2;
                                var node2X = node2Offset.left + node2.width() / 2;
                                var node2Y = node2Offset.top + node2.height() / 2;

                                // Inside the loop where you draw the lines
                                var weight = data.weights[i][j][k];

                                //if (k == 0) {
                                    //console.log(node1X, node1Y, node2X, node2Y, weight);
                                //}

                                drawLine(node1X, node1Y, node2X, node2Y, weight);
                            }
                        }
                    }
                });
        });
    }

    // Event listener for input changes
    $('.min_input, .max_input').on('input', function() {
        updateWeights();
    });

    var stop = false;
    var running = false;

    $('#start_training').on('click', function() {
        $('#start_training').css('display', 'none');
        $('#stop_training').css('display', 'block');

        // Get the value of the learning_rate input
        const learningRateValue = $('#learning_rate').val();
        console.log("Learning Rate:", learningRateValue)

        const batchSizeValue = $('#batch_size').val();
        console.log("Batch Size:", batchSizeValue)

        fetch('/nn/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    learning_rate: learningRateValue,
                    batch_size: batchSizeValue
                })
            })
            .then(response => response.json())
            .then(data => {
                console.log(data);
                if (!running) {
                    running = true;
                    var intervalId = setInterval(function() {
                        console.log("updating weights!");
                        updateWeights();

                        if (stop) {
                            stop = false;
                            running = false;
                            clearInterval(intervalId);
                        }
                    }, 3000);
                }
            });
    });

    $('#stop_training').on('click', function() {
        $('#start_training').css('display', 'block');
        $('#stop_training').css('display', 'none');
        fetch('/nn/stop')
            .then(response => response.json())
            .then(data => {
                console.log(data);
            });
        if (running) {
            stop = true;
        }
    });

    $('#new_nn').on('click', function() {
        $('#start_training').css('display', 'block');
        $('#stop_training').css('display', 'none');
        fetch('/nn/new')
            .then(response => response.json())
            .then(data => {
                console.log(data);
                updateWeights();
            });
        if (running) {
            stop = true;
        }
    });

    updateWeights();
}

function loadImage(node) {
    // Create a new PixiJS application
    var app = new PIXI.Application({
        width: 400, // Replace with your desired width
        height: 400, // Replace with your desired height
        backgroundColor: 0xffffff, // Set the background color
    });

    fetch('/api/images')
        .then(response => response.json())
        .then(images => {
            console.log(images);
            // Find the minimum and maximum weight values
            let minWeight = Math.min(...images.flat().filter(weight => !isNaN(weight)));
            let maxWeight = Math.max(...images.flat().filter(weight => !isNaN(weight)));
            let maxAbsWeight = Math.max(Math.abs(minWeight), Math.abs(maxWeight));
            //console.log("Maxes:", minWeight, maxWeight, maxAbsWeight);

            // Function to draw lines between nodes
            function determineColor(weight) {
                let lineColor = 'transparent'; // Default color is transparent

                if (weight !== 0) {
                    const opacity = Math.abs(weight) / maxAbsWeight; // Calculate opacity based on weight
                    //console.log("opacity", opacity,  Math.abs(weight), maxWeight);
                    lineColor = weight > 0 ? `rgba(0, 0, 255, ${opacity})` : `rgba(255, 0, 0, ${opacity})`; // Set color based on weight sign
                }

                return lineColor;
            }

            // Access the pixel values from your data
            var pixelValues = images[0][0]; // Replace with the actual pixel values from your data

            // Determine color for each pixel and draw on the Graphics object
            var graphics = new PIXI.Graphics();
            var pixelSize = 2; // Set the desired size for each pixel

            for (var i = 0; i < pixelValues.length; i++) {
                var pixelValue = pixelValues[i];
                var color = determineColor(pixelValue); // Implement your logic to determine the color based on the pixel value

                var x = i % 28 * pixelSize; // Replace 28 with the actual width of your image
                var y = Math.floor(i / 28) * pixelSize; // Replace 28 with the actual width of your image

                graphics.beginFill(color);
                graphics.drawRect(x, y, pixelSize, pixelSize);
                graphics.endFill();
            }

            // Add the Graphics object to the stage
            app.stage.addChild(graphics);

            // Render the PixiJS stage
            app.renderer.render(app.stage);

            window.requestAnimationFrame(() => {
                //console.log(data.weights.length);
                // ######## WEIGHTS ########
                for (var i = 0; i < data.weights.length; i++) {
                    for (var j = layer_ranges[i][0]-1; j < layer_ranges[i][1]; j++) {
                        for (var k = layer_ranges[i+1][0]-1; k < layer_ranges[i+1][1]; k++) {
                            var node1 = $('#node-' + i + '-' + j);
                            var node2 = $('#node-' + (i + 1) + '-' + k);

                            // Get positions of nodes
                            var node1Offset = node1.offset();
                            var node2Offset = node2.offset();

                            // Get coordinates of node centers relative to their parent containers
                            var node1X = node1Offset.left + node1.width() / 2;
                            var node1Y = node1Offset.top + node1.height() / 2;
                            var node2X = node2Offset.left + node2.width() / 2;
                            var node2Y = node2Offset.top + node2.height() / 2;

                            // Inside the loop where you draw the lines
                            var weight = data.weights[i][j][k];

                            //if (k == 0) {
                                //console.log(node1X, node1Y, node2X, node2Y, weight);
                            //}

                            drawLine(node1X, node1Y, node2X, node2Y, weight);
                        }
                    }
                }
            });
        });

}