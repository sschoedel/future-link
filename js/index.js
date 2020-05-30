
require([
  "esri/Map",
  "esri/views/MapView",
  "esri/views/ui/UI",
  "esri/Graphic",
  "esri/layers/GraphicsLayer"
], function(Map, MapView, UI, Graphic, GraphicsLayer) {

var map = new Map({
  basemap: "dark-gray-vector"
});

  var view = new MapView({
    container: "viewDiv",
    map: map,
    center: [-98.5795, 39.8283], // longitude, latitude
    zoom: 5,
    
  });

  view.ui.add([
    {
      component: "titleDiv",
      position: "manual"
    }
  ]);

  var graphicsLayer = new GraphicsLayer();
  map.add(graphicsLayer);  
  
  // Create a polygon geometry
  var polygon = {
   type: "polygon",
   rings: [
    [ -88.675628, 48.120444 ],
    [ -90.214866, 46.499947 ],
    [ -86.254996, 44.691935 ],
    [ -83.829224, 43.662632 ]
  ]
  };

  var simpleFillSymbol = {
   type: "simple-fill",
   color: [227, 139, 79, 0.8],  // orange, opacity 80%
   outline: {
     color: [255, 255, 255],
     width: 1
   }
  };

  var polygonGraphic = new Graphic({
   geometry: polygon,
   symbol: simpleFillSymbol
  });

  graphicsLayer.add(polygonGraphic);

});