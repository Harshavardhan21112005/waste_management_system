// // Initialize the map
// var map = L.map('map').setView([13.0827, 80.2707], 16);

// // Add OpenStreetMap tiles
// L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
//     maxZoom: 18,
//     attribution: '© OpenStreetMap'
// }).addTo(map);

// // Define the center point and 200m radius circle
// var centerLatLng = [13.0827, 80.2707];
// var radius = 200;
// var originalCenter = L.latLng(centerLatLng);

// // Draw a circle to represent the 200-meter radius
// var circle = L.circle(centerLatLng, {
//     color: '#4caf50',
//     fillColor: '#4caf50',
//     fillOpacity: 0.2,
//     radius: radius
// }).addTo(map);

// // Function to get new coordinates for random dustbin placement
// function getNewCoordinates(center, distance, angle) {
//     var angleInRadians = angle * (Math.PI / 180);
//     var newLat = center[0] + (distance / 111320) * Math.cos(angleInRadians);
//     var newLng = center[1] + (distance / (111320 * Math.cos(center[0] * (Math.PI / 180)))) * Math.sin(angleInRadians);
//     return [newLat, newLng];
// }

// // Function to place random dustbins based on integer array
// function placeRandomDustbins(integerArray) {
//     const minInterval = 30.48; // Minimum distance in meters (100 feet)
//     const maxInterval = 182.88; // Maximum distance in meters (600 feet)

//     for (let i = 0; i < integerArray.length; i++) {
//         const randomDistance = Math.random() * (maxInterval - minInterval) + minInterval;
//         const randomAngle = Math.random() * 360;
//         const binLocation = getNewCoordinates(centerLatLng, randomDistance, randomAngle);

//         var dustbinColor;
//         switch (integerArray[i]) {
//             case 0:
//                 dustbinColor = 'green'; // Not Full
//                 break;
//             case 1:
//                 dustbinColor = 'darkgreen'; // Empty Scattered
//                 break;
//             case 2:
//                 dustbinColor = 'red'; // Full
//                 break;
//             case 3:
//                 dustbinColor = 'orange'; // Full Scattered
//                 break;
//             default:
//                 dustbinColor = 'black'; // Default case
//                 break;
//         }

//         L.marker(binLocation, {
//             icon: L.divIcon({
//                 className: 'dustbin-icon',
//                 html: `<div style="background-color: ${dustbinColor}; width: 10px; height: 10px; border-radius: 50%;"></div>`,
//                 iconSize: [20, 20]
//             })
//         }).addTo(map).bindPopup("Dustbin: " + dustbinColor);
//     }
// }

// // Function to place a moving person marker
// function placeMovingPerson() {
//     const personRadius = 100; 
//     const randomDistance = Math.random() * personRadius;
//     const randomAngle = Math.random() * 360;
//     const personLocation = getNewCoordinates(centerLatLng, randomDistance, randomAngle);

//     L.marker(personLocation, {
//         icon: L.divIcon({
//             className: 'person-icon',
//             html: `<div style="background-color: blue; width: 15px; height: 15px; border-radius: 50%;"></div>`,
//             iconSize: [20, 20]
//         })
//     }).addTo(map).bindPopup("Person");
// }

// // Fetch the integer array and place markers
// fetch('results.txt')
//     .then(response => response.text())
//     .then(data => {
//         const integerArray = data.trim().split('').map(char => {
//             const number = parseInt(char, 10);
//             return !isNaN(number) ? number : null;
//         }).filter(num => num !== null);

//         // Place random dustbins and person marker
//         placeRandomDustbins(integerArray);
//         placeMovingPerson();
//     })
//     .catch(error => console.error('Error fetching file:', error));

// // Add recenter button
// L.control({ position: 'bottomleft' }).onAdd = function () {
//     var div = L.DomUtil.create('div', 'leaflet-control-custom recenter-btn');
//     div.innerHTML = '<button onclick="map.setView(originalCenter, 16)">Recenter</button>';
//     return div;
// }.addTo(map);

// // Add navigation buttons
// L.control({ position: 'topright' }).onAdd = function () {
//     var div = L.DomUtil.create('div', 'leaflet-control-custom compass-controls');
//     div.innerHTML = 
//         `<button onclick="map.panBy([0, -100])">North</button>
//         <button onclick="map.panBy([100, 0])">East</button>
//         <button onclick="map.panBy([-100, 0])">West</button>
//         <button onclick="map.panBy([0, 100])">South</button>`;
//     return div;
// }.addTo(map);


// Initialize the map
var map = L.map('map').setView([13.0827, 80.2707], 16);

// Add OpenStreetMap tiles
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 18,
    attribution: '© OpenStreetMap'
}).addTo(map);

// Define the center point and 200m radius circle
var centerLatLng = [13.0827, 80.2707];
var radius = 200;
var originalCenter = L.latLng(centerLatLng);

// Draw a circle to represent the 200-meter radius
var circle = L.circle(centerLatLng, {
    color: '#4caf50',
    fillColor: '#4caf50',
    fillOpacity: 0.2,
    radius: radius
}).addTo(map);

// Function to get new coordinates for random dustbin placement
function getNewCoordinates(center, distance, angle) {
    var angleInRadians = angle * (Math.PI / 180);
    var newLat = center[0] + (distance / 111320) * Math.cos(angleInRadians);
    var newLng = center[1] + (distance / (111320 * Math.cos(center[0] * (Math.PI / 180)))) * Math.sin(angleInRadians);
    return [newLat, newLng];
}

// Function to place random dustbins based on integer array
function placeRandomDustbins(integerArray) {
    const minInterval = 30.48; // Minimum distance in meters (100 feet)
    const maxInterval = 182.88; // Maximum distance in meters (600 feet)

    for (let i = 0; i < integerArray.length; i++) {
        const randomDistance = Math.random() * (maxInterval - minInterval) + minInterval;
        const randomAngle = Math.random() * 360;
        const binLocation = getNewCoordinates(centerLatLng, randomDistance, randomAngle);

        var dustbinColor;
        switch (integerArray[i]) {
            case 0:
                dustbinColor = 'green'; // Not Full
                break;
            case 1:
                dustbinColor = 'darkgreen'; // Empty Scattered
                break;
            case 2:
                dustbinColor = 'red'; // Full
                break;
            case 3:
                dustbinColor = 'orange'; // Full Scattered
                break;
            default:
                dustbinColor = 'black'; // Default case
                break;
        }

        L.marker(binLocation, {
            icon: L.divIcon({
                className: 'dustbin-icon',
                html: `<div style="background-color: ${dustbinColor}; width: 10px; height: 10px; border-radius: 50%;"></div>`,
                iconSize: [20, 20]
            })
        }).addTo(map).bindPopup("Dustbin: " + dustbinColor);
    }
}

// Function to place a moving person marker
function placeMovingPerson() {
    const personRadius = 100; 
    const randomDistance = Math.random() * personRadius;
    const randomAngle = Math.random() * 360;
    const personLocation = getNewCoordinates(centerLatLng, randomDistance, randomAngle);

    L.marker(personLocation, {
        icon: L.divIcon({
            className: 'person-icon',
            html: `<div style="background-color: blue; width: 15px; height: 15px; border-radius: 50%;"></div>`,
            iconSize: [20, 20]
        })
    }).addTo(map).bindPopup("Person");
}

// Fetch the integer array and place markers
fetch('results.txt')
    .then(response => response.text())
    .then(data => {
        const integerArray = data.trim().split('').map(char => {
            const number = parseInt(char, 10);
            return !isNaN(number) ? number : null;
        }).filter(num => num !== null);

        // Place random dustbins and person marker
        placeRandomDustbins(integerArray);
        placeMovingPerson();
    })
    .catch(error => console.error('Error fetching file:', error));

// Add recenter button
L.control({ position: 'bottomleft' }).onAdd = function () {
    var div = L.DomUtil.create('div', 'leaflet-control-custom recenter-btn');
    div.innerHTML = '<button onclick="map.setView(originalCenter, 16)">Recenter</button>';
    return div;
}.addTo(map);

// Add navigation buttons
L.control({ position: 'topright' }).onAdd = function () {
    var div = L.DomUtil.create('div', 'leaflet-control-custom compass-controls');
    div.innerHTML = `
        <button onclick="map.panBy([0, -100])">North</button>
        <button onclick="map.panBy([100, 0])">East</button>
        <button onclick="map.panBy([-100, 0])">West</button>
        <button onclick="map.panBy([0, 100])">South</button>
    `;
    return div;
}.addTo(map);