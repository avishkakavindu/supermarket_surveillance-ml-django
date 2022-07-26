$(document).ready(function () {

    const webcamElement = document.getElementById('webcam');
    const canvasElement = document.getElementById('img_canvas');
    const webcam = new Webcam(webcamElement, 'user', canvasElement);


    webcam.start()
        .then(result => {
            console.log("webcam started");
        })
        .catch(err => {
            console.log(err);
        });
});
