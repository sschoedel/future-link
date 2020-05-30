require([
    "esri/Map",
    "esri/views/MapView",
    "esri/views/ui/UI"
  ], function(Map, MapView, UI) {

  var map = new Map({
    basemap: "dark-gray-vector"
  });

  var view = new MapView({
    container: "viewDiv",
    map: map,
    center: [-98.5795, 39.8283], // longitude, latitude
    zoom: 4,
    
  });


  view.ui.add([
    {
      component: "titleDiv",
      position: "manual"
    }
  ]);
});

