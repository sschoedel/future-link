require([
    "esri/Map",
    "esri/views/MapView"
  ], function(Map, MapView) {

  var map = new Map({
    basemap: "dark-gray-vector"
  });

  var view = new MapView({
    container: "viewDiv",
    map: map,
    center: [-98.5795, 39.8283], // longitude, latitude
    zoom: 5
  });
});

