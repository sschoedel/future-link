import xml.etree.ElementTree as ET
tree = ET.parse('states.xml')
root = tree.getroot()

for state in root:

	# Polygon outline
	print('var {} = {{'.format(state.attrib['name'].replace(" ","")))
	print('  type: "polygon",')
	print('  rings: [')
	# Print all coordinates
	length = len(state)
	for i, point in enumerate(state):
		if i != length-1:
			print('[{}, {}],'.format(point.attrib['lng'], point.attrib['lat']), end =" ")
		else:
			print('[{}, {}]'.format(point.attrib['lng'], point.attrib['lat']))
	print('  ]')
	print('};')
	print()

	# Fill symbol
	print('var simpleFillSymbol{} = {{'.format(state.attrib['name'].replace(" ","")))
	print('  type: "simple-fill",')
	print('  color: [227, 139, 79, 0.8],') # Orange
	print('  outline: {')
	print('    color: [255, 255, 255],')
	print('    width: 1')
	print('  }')
	print('};')
	print()

	# Graphic
	print('var polygonGraphic{} = new Graphic({{'.format(state.attrib['name'].replace(" ","")))
	print('  geometry: {},'.format(state.attrib['name'].replace(" ","")))
	print('  symbol: simpleFillSymbol{}'.format(state.attrib['name'].replace(" ","")))
	print('})')
	print()

	# Add graphics layer
	print('graphicsLayer.add(polygonGraphic{});'.format(state.attrib['name'].replace(" ","")))
	print()
	print()