//radius of click around the first point to close the draw
var END_CLICK_RADIUS = 15;
//the max number of points of your polygon
var MAX_POINTS = 20;

var mouseX = 0;
var mouseY = 0;
var isStarted = false;
var polygons = [];

var canvas = null;
var ctx;
var image;
var steps = [];
var isPolygon = false;
var didClick = false;
var current_user_name = '';
var current_image_name = '';

var current_img = "";
var is_cam_on = false;
var is_invert = false;
var is_zoomed = false;

var polygonColor = 'red';
var color = 'red';

var canvasWidth = 0;
var canvasHeight = 0;

var newDraw = false;
var topInterval, rightInterval, leftInterval, bottomInterval;

var isCircle = false;
var circle = [];
var polygonColor = 'red';


$(document).ready(function () {

	if (sessionStorage.getItem('username') && sessionStorage.getItem('password')){
		current_user_name = sessionStorage.getItem('username')
	} else {
		window.open('/ai/login/', '_self');
	}

	
   
    $(document).on('keydown', function(e) {
    switch (e.keyCode) {
        case 37:
            pan('left');
            break;
        case 38:
            pan('top');
            break; 
        case 39:
            pan('right');
            break; 
        case 40:
            pan('bottom');
            break;
    }
});

	// Set shortcuts
	$(document).keypress(function(event) {
		
		if(String.fromCharCode(event.which).toLowerCase() == "a") {
			clicked(0);
		}
		else if(String.fromCharCode(event.which).toLowerCase() == "n") {
			clicked(1);
		}
		else if(String.fromCharCode(event.which).toLowerCase() == "i") {
			invertColors();
		}
		else if(String.fromCharCode(event.which).toLowerCase() == "z") {
			zoomImg()
		}
		else if(String.fromCharCode(event.which).toLowerCase() == "c") {
			switchImage();
		}
		else if(event.which == 32)	//space bar
		{
			event.preventDefault();
			clicked(2);
		}
	});

	$("#loading").show();
	$("#annotation_image").hide();
	$("#wrapper").css("max-height", $("#wrapper").height());
	$("#wrapper").css("max-width", $("#wrapper").width());

	// Initializing canvas and draw color
	canvas = document.getElementById("annotation_image");
	ctx = canvas.getContext("2d");
	image = new Image();
	image.src = document.querySelector('canvas').dataset.image='';

	$("#annotation_image").panzoom({
		$zoomIn: $(".zoom-in"),
		$zoomOut: $(".zoom-out"),
		$reset: $(".reset"),
		increment: 0.05,
	});

	$("#annotation_image").panzoom().on('panzoomzoom', function(e, panzoom, scale, opts) {
		if (isPolygon) {
			togglePolygon(polygonColor);
		}
		if (isCircle) {
            toggleCircle(circleColor);
        }

	});

	canvas.addEventListener("click", function (e) {
		var x, y;

		var values = applyOffset(e.pageX, e.pageY);
		x = values['x'];
		y = values['y'];


		if(isPolygon) {
			if (isStarted) {
				//drawing the next line, and closing the polygon if needed
				if (Math.abs(x - polygons[polygons.length - 1][0].x) < END_CLICK_RADIUS && Math.abs(y - polygons[polygons.length - 1][0].y) < END_CLICK_RADIUS) {
					togglePolygon(polygonColor);
					isStarted = false;
                    newDraw = true;
                    local_data = {'file_name':current_image_name, 'username':current_user_name,'cords':JSON.stringify(polygons)};
					localStorage.setItem(current_image_name, JSON.stringify(local_data))
				} else {

					polygons[polygons.length - 1].push(new Point(x, y, polygonColor, canvas.width, canvas.height));
					if (polygons[polygons.length - 1].length >= MAX_POINTS) {
						togglePolygon(polygonColor);
						isStarted = false;
                        newDraw = true;
    					local_data = {'file_name':current_image_name, 'username':current_user_name,'cords':JSON.stringify(polygons)};
                        localStorage.setItem(current_image_name, JSON.stringify(local_data))

					}
				   
				   
				}
			} else {
				//opening the polygon
				polygons.push([new Point(x, y, polygonColor,canvas.width, canvas.height)]);

				isStarted = true;
			}
		}
		if (isCircle) {
            if (isStarted) {
                circle[circle.length - 1].x2 = x;
                circle[circle.length - 1].y2 = y;
                isStarted = false;
                toggleCircle(circleColor);
               // local_data = { 'file_name': current_image_name, 'username': current_user_name,  'cords': JSON.stringify(circle)}
               //  // local_data = {'file_name':current_image_name, 'username':current_user_name,'circle':JSON.stringify(circle)};
               //  localStorage.setItem(current_image_name, JSON.stringify(local_data))
            	saveCircles(circle);
            } else {
               	circle.push({ 'x1': x, 'y1': y, 'color': circleColor, 'canvasWidth': canvas.width, 'canvasHeight': canvas.height });
                isStarted = true;
            }
        }

	}, false);

	//we just save the location of the mouse
	canvas.addEventListener("mousemove", function (e) {
	   
		var values = applyOffset(e.pageX, e.pageY);
		mouseX = values['x'];
		mouseY = values['y'];
	}, false);

	get_next_path();

	// Set interval for refreshing canvas
	setInterval("draw();", 5);
});

// Wait for window to load before setting canvas size
$(window).load(function() {
	// Set canvas size
	canvas.width = $("#annotation_image").width() - 20;
	canvas.height = $("#annotation_image").height();

	canvas.offsetLeft = 0;
	canvas.offsetTop = 0;

	console.log("Canvas", $("#annotation_image").width(), canvas.width);
});

var applyOffset = function(x, y) {

	var transformMatrix = $("#annotation_image").panzoom('getMatrix');
	// console.log("Transform - ", transformMatrix);
	y -= parseInt(transformMatrix[5]);
	x -= parseInt(transformMatrix[4]);
		// Plus 20 to compensate for margins
	x -= $("#left-bar").width() + 20;
	for (var i = 0; i < steps.length; i++) {

		switch (steps[i]) {
			case "zoomIn":
				x *= (1/1.05);
				y *= (1/1.05);
				break;
			case "zoomOut":
				x *= 1.05;
				y *= 1.05;
				break;
			case "panLeft":
				x += 20;
				break;
			case "panRight":
				x -= 20;
				break;
			case "panTop":
				y += 20;
				break;
			case "panBottom":
				y -= 20;
				break;
		}
	}
	return {
		"x": x,
		"y": y
	}
};

//object representing a point
function Point(x, y, polygonColor, canvasWidth, canvasHeight) {
    this.x = x;
    this.y = y;
    this.color = polygonColor;
    this.canvasWidth = canvasWidth
    this.canvasHeight = canvasHeight;
}


function togglePolygon(color) {
	var redPencil = document.querySelector("#redPencil");
	var greenPencil = document.querySelector("#greenPencil");
	//var pencil = document.querySelector('#pencil' + color);
	//polygonColor = color;
	var transformMatrix = $("#annotation_image").panzoom('getMatrix');
	if (!isPolygon) {
		
		if(transformMatrix[0].toString() !== "1" || transformMatrix[3].toString() !== "1" || transformMatrix[4].toString() !== "0" || transformMatrix[5].toString() !== "0") {
			alert('Please reset the zoom to draw. Zoom can be reset by clicking the last icon in the bar.');
		} else if(steps.length > 0){
			 alert('Please reset the zoom to draw. Zoom can be reset by clicking the last icon in the bar.');

		}

		else 
		{
			$("#annotation_image").panzoom("disable");
			isPolygon = !isPolygon;
			if(color == 'red')
			{
				$("#redPencil").attr('src', '/static/assets/img/red_pencil.png');
				$("#greenPencil").attr('src', '/static/assets/img/normal_pencil.png');
				// redPencil.style.color = color;
				// greenPencil.style.color = 'black';
			}
			else{

				$("#redPencil").attr('src', '/static/assets/img/normal_pencil.png');
				$("#greenPencil").attr('src', '/static/assets/img/green_pencil.png');
				// redPencil.style.color = 'black';

				// greenPencil.style.color = color;
			}
		}
	} else {
		isPolygon = !isPolygon;
		
		$("#annotation_image").panzoom("enable");
		if(polygonColor == color)
		{
			$("#redPencil").attr('src', '/static/assets/img/normal_pencil.png');
			$("#greenPencil").attr('src', '/static/assets/img/normal_pencil.png');
		}
		else
		{
			if(color == 'red')
			{
				$("#redPencil").attr('src', '/static/assets/img/red_pencil.png');
				$("#greenPencil").attr('src', '/static/assets/img/normal_pencil.png');

				// redPencil.style.color = color;
				// greenPencil.style.color = 'black';
			}
		   else
		   {
		   	$("#redPencil").attr('src', '/static/assets/img/normal_pencil.png');
				$("#greenPencil").attr('src', '/static/assets/img/green_pencil.png');
				// redPencil.style.color = 'black';
				// greenPencil.style.color = color;
		   }
		}
	}

	polygonColor = color;
	
}

//resets the application
function reset() {
	isStarted = false;
    newDraw = false;
	polygons = [];
	circle = [];
}


function removeLastPolygon(){

    if(polygons.length > 0)
    {
        polygons.splice(-1,1);
        newDraw = true;
        localStorage.clear();
        local_data = {'file_name':current_image_name, 'username':current_user_name,'cords':JSON.stringify(polygons)};
        localStorage.setItem(current_image_name, JSON.stringify(local_data))

    }
}

function save() {
	var newArray = [];
	polygons.forEach(function(polygon) {
		newArray.push(JSON.stringify(polygon));
	});
}

//draws the current shape
function draw() {
    clear();
    // ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

    var top = 0;
    var left = 0;
    var width = image.naturalWidth;
    var height = image.naturalHeight;

    if ((width >= height) || (image.naturalWidth/image.naturalHeight < 1)) {
        
        width = canvas.width;
        height = image.naturalHeight/image.naturalWidth * canvas.width;
        top = (canvas.height - height) / 2;
            
    } else {
        
        height = canvas.height;
        width = image.naturalWidth/image.naturalHeight * canvas.height;
        left = (canvas.width - width) / 2;
            
    }

    ctx.drawImage(image, left, top, width, height); // render original image
    // console.log("Polygons in draw - ", polygons);
    polygons.forEach(function (points, i) {

        ctx.beginPath();

        if (points != null) {
        	console.log(points)
            points.forEach(function (p, j) {
                
                nw = p.canvasWidth / canvas.width;
                nh = p.canvasHeight / canvas.height;
                // console.log(nw, nh)
                newX = p.x / nw;
                newY = p.y / nh;


              
                if (j) {
                    ctx.lineTo(newX, newY);
                } else {
                    ctx.moveTo(newX, newY);
                }
                ctx.strokeStyle = points[j].color;
            });
        }
        if (i + 1 === polygons.length && isStarted) { // just the last one
            ctx.lineTo(mouseX, mouseY);
        } else {
            newX = points[0].x / nw;
            newY = points[0].y / nh;
            ctx.lineTo(newX, newY);
        }

        ctx.lineWidth = 5;
        ctx.stroke();
        // ctx.closePath();
        // ctx.fillStyle = "red";
        // ctx.fill();
    });

    createCircle();
}


// function saveResult() {
// 	var newCords = [];

// 	polygons.forEach(function(polygon) {
// 		newCords.push(JSON.stringify(polygon));
// 	});
// 	var data = { 'file_name': current_image_name, 'username': current_user_name, 'annotation_result': annotation_result, 'cords': JSON.stringify(circle) };
// 	console.log("Circles")
// 	console.log(circle)

	// var data = { 'file_name': 'demoimg.jpg', 'username': 'admin', 'annotation_result': true, 'cords': JSON.stringify(polygons) };
	// $.ajax({
	// 	url: '/ai/annotationDataPost/',
	// 	type: 'post',
	// 	data: data,
	// 	dataType: 'json',
	// 	success: function (resp) {
	// 		if (resp.message === 'saved') {
	// 			forward()
	// 		}
	// 	}
	// });
// }

function clear() {
	ctx.clearRect(0, 0, image.naturalWidth, image.naturalHeight);
}

function render() {
	clear();
	ctx.drawImage(image, 0, 0, image.naturalWidth, image.naturalHeight); // render original image
}

// function zoomIn() {
//     clear();
//     // ctx.translate(canvas.width * 0.5, canvas.height * 0.5);  // pivot = center
//     ctx.scale(1.05, 1.05);
//     // ctx.translate(-canvas.width * 0.5, -canvas.height * 0.5);
//     steps.push("zoomIn");
//     render();
// }

// function zoomOut() {
//     clear();
//     // ctx.translate(canvas.width * 0.5, canvas.height * 0.5);
//     ctx.scale((1/1.05), (1/1.05));
//     // ctx.translate(-canvas.width * 0.5, -canvas.height * 0.5);
//     steps.push("zoomOut");
//     render();
// }

function pan(direction) {
	
	
     if (isPolygon) {
        togglePolygon(polygonColor);
     }

     if (isCircle) {
        toggleCircle(circleColor);
    }


    var transformMatrix = $("#annotation_image").panzoom('getMatrix');
    // console.log("Transform - ", transformMatrix);
    y = parseInt(transformMatrix[5]);
    x = parseInt(transformMatrix[4]);
	clear();
    switch (direction) {
		case 'left':
			// ctx.translate(-20, 0);
            $("#annotation_image").panzoom("pan", x-20, y);

			steps.push("panLeft");
			break;
		case 'right':
			// ctx.translate(20, 0);
            $("#annotation_image").panzoom("pan", x+20, y);

			steps.push("panRight");
			break;
		case 'top':
			// ctx.translate(0,-20);
            $("#annotation_image").panzoom("pan", x, y-20);

			steps.push("panTop");
			break;
		case 'bottom':
			// ctx.translate(0, 20);
            $("#annotation_image").panzoom("pan", x, y+20);

			steps.push("panBottom");
			break;
	}
	render();
}

function restore() {
	render();
}

function clicked(annotation_result) {
	var new_cords = [];
	didClick = true;
	// polygons.forEach(function(polygon) {
	// 	new_cords.push(JSON.stringify(polygon));
	// });
	// local_data = {'file_name':current_image_name, 'username':current_user_name, 'annotation_result':annotation_result, 'cords':JSON.stringify(polygons)};
	// localStorage.setItem(current_image_name, JSON.stringify(local_data))
	data = { 'file_name': current_image_name, 'username': current_user_name, 'annotation_result': annotation_result, 'cords': JSON.stringify(circle)}

	$.ajax({
		url: '/ai/annotationDataPost/',
		type: 'post',
		dataType: 'json',
		success: function (resp) {
			if(resp.message == 'saved'){
                localStorage.clear();
                newDraw = false;
				forward()
			}
		},
		data: data
	});
}


function setPolygonsLocal(){
	var fileData = JSON.parse(localStorage.getItem(current_image_name));

    
	if(fileData!=null)
	{
		if(current_image_name == fileData.file_name)
        {
     		polygons = JSON.parse(fileData.circle);
          
        }
	}
}

function setCirclesLocal(){
	var fileData = JSON.parse(localStorage.getItem(current_image_name));

    console.log(fileData)
	if(fileData!=null)
	{
		if(current_image_name == fileData.file_name)
        {
     		polygons = JSON.parse(fileData.cords);
          
        }
	}
}
function setPolygons(fileData) {
    // console.log("File Data");
    // console.log(fileData);
    if (fileData !== null) {
        // if (current_image_name == fileData.file_name) {
        //     polygons = JSON.parse(fileData.cords);
        // }
        polygons = fileData;

    local_data = {'file_name':current_image_name, 'username':current_user_name, 'cords':JSON.stringify(polygons)};
    localStorage.setItem(current_image_name, JSON.stringify(local_data))

    }
}

function setCircles(circleData) {
    // console.log(circleData);
    if (circleData !== null) {
        // if (current_image_name == fileData.file_name) {
        //     polygons = JSON.parse(fileData.cords);
        // }
        circle = circleData;
     //    local_data = {'file_name':current_image_name, 'username':current_user_name, 'cords':JSON.stringify(polygons)};
    	// localStorage.setItem(current_image_name, JSON.stringify(local_data))
    }
}



function back() {

    if(newDraw)
    {
        alert("Looks like you have drawn more polygons. Please annotate to continue.");
    }
    else{

        localStorage.clear();
    	current_user_name = sessionStorage.getItem("username")
    	$.ajax({
    		url: '/ai/getPrevPath/?username='+current_user_name+'&current_img='+current_image_name,
    		type: 'get',
    		success: function (resp) {
                
    			if (resp.path !== ''){

    				$("#wrapper").show();
    				$(".annotation-btn").show();
    				$("#annotation").show();

    				if(is_cam_on) {
    					switchImage();
    				}
    				current_image_name = resp.file_name

    				zoomImg();
    				if(is_invert) {
    					invertColors();
    				}

    				image = new Image();
    				image.src = document.querySelector('canvas').dataset.image = resp.path;
                    localStorage.clear();
    				reset();
    				localStorage.setItem('circle', JSON.stringify(resp.cords));
    				setCircles(resp.cords);

    				render();
    				$('#annotation_image').panzoom("reset");
    				$('#annotation_image').width('100%').height('100%');

    				$("#counter").text(resp.current_index + " of " + resp.total_files);
    				// $("#annotation").css('color', "#ffffff");
    				if(resp.annotation == "0")
    				{
    					$("#annotation").text("Abnormal");
    					$("#annotation").css("color", "#c9302c");
    				}
    				else if(resp.annotation == "1")
    				{
    					$("#annotation").text("Normal");
    					$("#annotation").css("color", "#449d44");
    				}
    				else if(resp.annotation == "2")
    				{
    					$("#annotation").text("Not clear");
    					
    					$("#annotation").css("color", "#286090");
    				}
    				else{
    					$("#annotation").text("N/A");
    					$("#annotation").css("color", "black");
    				
    				}
    			}
    			else {
    				if(resp.current_index >= 1) {
    					$("#counter").text(resp.current_index + " of " + resp.total_files);
    				}
    			}
    		},
    	});
    }
}

function forward() {
   
	var annote_text =   $("#annotation").text();
    var fileData = JSON.parse(localStorage.getItem(current_image_name));
    localStorage.clear();
	if(annote_text == "N/A" && !didClick)
	{
		alert("Please annotate the image before going to the next image.");
	}
    else if(newDraw)
    {
        alert("Looks like you have drawn more polygons. Please annotate to continue.");
    }
	else
	{
		current_user_name = sessionStorage.getItem("username");
		$.ajax({
			url: '/ai/getForwardPath/?username='+current_user_name+'&current_img='+current_image_name,
			type: 'get',
			success: function (resp) {
				if (resp.path != ''){

					$("#wrapper").show();
					$(".annotation-btn").show();
					$("#annotation").show();

					 if(is_cam_on) {
						switchImage();
					}

					current_image_name = resp.file_name

					zoomImg();

					if(is_invert) {
						invertColors();
					}

					image = new Image();
					image.src = document.querySelector('canvas').dataset.image = resp.path;
                    localStorage.clear();

                    canvas.width = canvas.offsetWidth;
                	canvas.height = canvas.offsetHeight;
					reset();
                    // setPolygonsLocal();
					// setPolygons(resp.cords);
					localStorage.setItem('circle', JSON.stringify(resp.cords));
					setCircles(resp.cords);
					// setCircleLocal();

					render();
					$('#annotation_image').width('100%').height('100%');
				   

					$("#counter").text(resp.current_index + " of " + resp.total_files);
					// $("#annotation").css('color', "#ffffff");
					if(resp.annotation == "0")
    				{
    					$("#annotation").text("Abnormal");
    					$("#annotation").css("color", "#c9302c");
    				}
    				else if(resp.annotation == "1")
    				{
    					$("#annotation").text("Normal");
    					$("#annotation").css("color", "#449d44");
    				}
    				else if(resp.annotation == "2")
    				{
    					$("#annotation").text("Not clear");
    					
    					$("#annotation").css("color", "#286090");
    				}
    				else{
    					$("#annotation").text("N/A");
    					$("#annotation").css("color", "black");
    				
    				}
				} else {
					
					// $("#annotation").css('color', "#ffffff");
					if(resp.current_index == resp.total_files) 
					{
						$("#wrapper").hide();
						$(".annotation-btn").hide();
						$("#annotation").hide();
						current_image_name = '';
						alert("No more images left to annotate");
						$("#counter").text(resp.current_index + " of " + resp.total_files);
					}

					$("#annotation").text("N/A");
				}
			}
		});
	}
}

function get_next_path() {
	current_user_name = sessionStorage.getItem("username");
	$.ajax({
		url: '/ai/getNextPath/?username='+current_user_name,
		type: 'get',
		success: function (resp) {
			if(resp.status_code === 400) {
				alert("User not found.");
			}
			else if (resp.path !== '') {
				current_image_name = resp.file_name
				$("#loading").hide();
				$("#annotation_image").show();

				image = new Image();
				image.src = document.querySelector('canvas').dataset.image = resp.path;
				localStorage.clear();
				reset();
            
				 // Imp:- Pass css sizes to canvas to avoid misplacement of points
                canvas.width = canvas.offsetWidth;
                canvas.height = canvas.offsetHeight;
				render();
				$('#annotation_image').width('100%').height('100%');
				$("#annotation").text("N/A");
				$("#annotation").css('color', "black");
				$("#counter").text(resp.current_index + " of " + resp.total_files);

			} else {
				$("#counter").text(resp.current_index + " of " + resp.total_files);
				if(resp.current_index === resp.total_files) {
					$("#wrapper").hide();
					$(".annotation-btn").hide();
					current_image_name = '';
					$("#annotation").hide();
				}
				alert("No more images left to annotate");
			}
		},
		error: function(e) {
			alert("You are not authorized to annotate images. Please contact the administrator");
		}
	});
}

function invertColors() {
	if(document.getElementById("annotation_image").style.filter === "invert(0%)" || document.getElementById("annotation_image").style.filter === "" ) {
		document.getElementById("annotation_image").style.filter = "invert(100%)";
		is_invert = true;
	} else {
		document.getElementById("annotation_image").style.filter = "invert(0%)";
		is_invert = false;
	}
}

function zoomImg() {
	ctx.resetTransform();
	$('#annotation_image').panzoom("reset");
	steps = []
}

function switchImage() {

	if(!is_cam_on) {
		current_img = current_image_name
		cam_image_path = "input_image_cam_"+current_img.substring(current_img.indexOf("/")+1);

		image = new Image();
		image.src = document.querySelector('canvas').dataset.image = "/static/input_files/"+cam_image_path
		is_cam_on = true
	} else {
		image.src = document.querySelector('canvas').dataset.image =  "/static/media/"+current_image_name
		is_cam_on = false
	}
	reset();
	console.log("Hello");
	console.log(localStorage.getItem('circle'));
	setCircles(JSON.parse(localStorage.getItem('circle')));
	// setPolygonsLocal();
	render();
}

function drawPoint(x, y, color) {
    ctx.beginPath();
    ctx.arc(x-10,y-20,2,0,2*Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
}


function createCircle() {
   circle.forEach(function (element, index) {
        if (isStarted && (index === circle.length-1)) {
            drawPoint(element.x1, element.y1, element.color);
        }


        // calculation for different screens
        nw = element.canvasWidth / canvas.width;
        nh = element.canvasHeight / canvas.height;
        newX1 = element.x1 / nw - 10;
        newY1 = element.y1 / nh - 20;
        newX2 = element.x2 / nw - 10;
        newY2 = element.y2 / nh - 20;


        ctx.strokeStyle = element.color;
        ctx.beginPath();
        // ctx.moveTo(element.x1 + getRadius(element.x1, element.x2, element.y1, element.y2), element.y1);
        ctx.arc(newX1, newY1, getRadius(newX1, newX2, newY1, newY2), 0, 2*Math.PI);
        ctx.lineWidth = 5;
        ctx.closePath();
        ctx.stroke();
    });
}


function getRadius(x1, x2, y1, y2) {
    return Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
}


function toggleCircle(color) {

    var redPencil = document.querySelector('#redPencil');
    var greenPencil = document.querySelector('#greenPencil');
    var transformMatrix = $("#annotation_image").panzoom('getMatrix');

    if (!isCircle) {
        if (transformMatrix[0].toString() !== "1" || transformMatrix[3].toString() !== "1" || transformMatrix[4].toString() !== "0" || transformMatrix[5].toString() !== "0") {
            alert('You can only draw without zoom/pan');
        } else {
            isCircle = !isCircle;
            if(color === 'green')
			{
				$("#redPencil").attr('src', '/static/assets/img/green_pencil.png');
				$("#greenPencil").attr('src', '/static/assets/img/normal_pencil.png');
				// redPencil.style.color = color;
				// greenPencil.style.color = 'black';
			}
			else{

				$("#redPencil").attr('src', '/static/assets/img/normal_pencil.png');
				$("#greenPencil").attr('src', '/static/assets/img/red_pencil.png');
				// redPencil.style.color = 'black';

				// greenPencil.style.color = color;
			}

        }
    } else {
        if (circleColor === color) {
            isCircle = !isCircle;
            $("#greenPencil").attr('src', '/static/assets/img/normal_pencil.png');
            $("#redPencil").attr('src', '/static/assets/img/normal_pencil.png');
            // redPencil.style.color = 'black';
            // greenPencil.style.color = 'black';
        } else {
           if(color === 'green')
			{
				$("#redPencil").attr('src', '/static/assets/img/green_pencil.png');
				$("#greenPencil").attr('src', '/static/assets/img/normal_pencil.png');
				// redPencil.style.color = color;
				// greenPencil.style.color = 'black';
			}
			else{

				$("#redPencil").attr('src', '/static/assets/img/normal_pencil.png');
				$("#greenPencil").attr('src', '/static/assets/img/red_pencil.png');
				// redPencil.style.color = 'black';

				// greenPencil.style.color = color;
			}
        }

    }

    circleColor = color;
}

function saveCircles(data) {
    localStorage.setItem('circle', JSON.stringify(data));
}