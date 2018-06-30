$(document).ready(function(){
  //$('.tabs').tabs({ 'swipeable': true, 'responsiveThreshold': Infinity });
  $('.tabs').tabs();

  var coll = document.getElementsByClassName("collapse");
  var i;
  for (i = 0; i < coll.length; i++) {
    coll[i].addEventListener("click", function() {
      var content = this.nextElementSibling;
      if (content.style.display === "block") {
        content.style.display = "none";
      } else {
        content.style.display = "block";
      }
    });
  }
});
