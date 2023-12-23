var socket = io.connect(
  "http://" + location.hostname + ":" + location.port + "/"
);

socket.on("update", function (data) {
  document.getElementById("score").innerText = data.score;
  document.getElementById("flag").innerText = data.flag ? "Alert" : "Drowsy";
  document.getElementById("alerts").innerText = data.flag
    ? "None"
    : "Drowsiness Detected";
  if (!data.flag) {
    document.getElementById("video_feed").style.border = "10px solid red";
  } else {
    document.getElementById("video_feed").style.border = "none";
  }
});
