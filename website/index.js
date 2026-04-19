const HK_CENTER = [21, 112];
const HK_BOUNDS = [
	[21.13, 112.78],
	[23.58, 115.53],
];

const FORECAST_DATA_URL = "forcast_data/predicted_traffic_speed.json";

const PLAY_INTERVAL_MS = 500;
const FORECAST_REFRESH_INTERVAL_MS = 60 * 1000 * 5;

const els = {
	btnBackward: document.getElementById("btnBackward"),
	btnPlay: document.getElementById("btnPlay"),
	btnForward: document.getElementById("btnForward"),
	mapStyleSelect: document.getElementById("mapStyleSelect"),
	timeSlider: document.getElementById("timeSlider"),
	labelMinTime: document.getElementById("labelMinTime"),
	labelMaxTime: document.getElementById("labelMaxTime"),
	currentPlayingTime: document.getElementById("currentPlayingTime"),
	realLifeTime: document.getElementById("realLifeTime"),
};

const map = L.map("map", {
	center: HK_CENTER,
	zoom: 13,
	minZoom: 11.4,
	maxZoom: 17,
	maxBounds: HK_BOUNDS,
	maxBoundsViscosity: 1.0,
	zoomControl: false,
	preferCanvas: true,
});

const tileLayerConfigs = {
	dark: {
		url: "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
		options: {
			maxZoom: 20,
			subdomains: "abcd",
			attribution: "&copy; OpenStreetMap contributors &copy; CARTO",
		},
	},
	light: {
		url: "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
		options: {
			maxZoom: 20,
			subdomains: "abcd",
			attribution: "&copy; OpenStreetMap contributors &copy; CARTO",
		},
	},
	voyager: {
		url: "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
		options: {
			maxZoom: 20,
			subdomains: "abcd",
			attribution: "&copy; OpenStreetMap contributors &copy; CARTO",
		},
	},
	osm: {
		url: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
		options: {
			maxZoom: 19,
			attribution: "&copy; OpenStreetMap contributors",
		},
	},
};

let activeBaseLayer = null;

function applyMapStyle(styleKey) {
	const nextConfig = tileLayerConfigs[styleKey] || tileLayerConfigs.dark;
	if (activeBaseLayer) {
		map.removeLayer(activeBaseLayer);
	}

	activeBaseLayer = L.tileLayer(nextConfig.url, nextConfig.options).addTo(map);
}

function bindMapStyleControl() {
	if (!els.mapStyleSelect) {
		applyMapStyle("dark");
		return;
	}

	applyMapStyle(els.mapStyleSelect.value || "dark");
	els.mapStyleSelect.addEventListener("change", () => {
		applyMapStyle(els.mapStyleSelect.value);
	});
}

map.fitBounds(HK_BOUNDS);

const renderer = L.canvas({ padding: 0.5 });
const detectorMarkers = [];
let forecastTimestamps = [];
let forecastSpeedsByDetector = new Map();

let timelineIndex = 0;
let isPlaying = false;
let playTimer = null;
let forecastRefreshTimer = null;
let isRefreshingForecast = false;

function speedToColor(speed) {

	if (!Number.isFinite(speed)) return "#6c7a86";
	if (speed < 20) return "#d01c1f";
	if (speed < 35) return "#ff8b00";
	if (speed < 50) return "#f2cb05";
	if (speed < 70) return "#55ba3a";
	return "#0d8fe8";

}

function formatDateTime(dateObj) {
	const pad = (n) => String(n).padStart(2, "0");
	return `${dateObj.getFullYear()}-${pad(dateObj.getMonth() + 1)}-${pad(dateObj.getDate())} ${pad(
		dateObj.getHours()
	)}:${pad(dateObj.getMinutes())}`;
}

function parseForecastTimestamp(timestamp) {
	if (!timestamp || timestamp.length !== 14) {
		return null;
	}

	const year = Number(timestamp.slice(0, 4));
	const month = Number(timestamp.slice(4, 6)) - 1;
	const day = Number(timestamp.slice(6, 8));
	const hour = Number(timestamp.slice(8, 10));
	const minute = Number(timestamp.slice(10, 12));
	const second = Number(timestamp.slice(12, 14));
	const date = new Date(year, month, day, hour, minute, second);

	return Number.isNaN(date.getTime()) ? null : date;
}

function formatForecastLabel(timestamp) {
	const parsed = parseForecastTimestamp(timestamp);
	return parsed ? formatDateTime(parsed) : "--";
}

function parseCsv(text) {
	const rows = text
		.split(/\r?\n/)
		.map((line) => line.trim())
		.filter((line) => line.length > 0);

	if (rows.length < 2) {
		return [];
	}

	const headers = rows[0].split(",").map((h) => h.replace(/^\uFEFF/, "").trim());

	return rows.slice(1).map((line) => {
		const cols = line.split(",");
		const item = {};
		headers.forEach((header, i) => {
			item[header] = (cols[i] || "").trim();
		});
		return item;
	});
}

function updatePlayButton() {
	els.btnPlay.textContent = isPlaying ? "❚❚" : "▶";
}

function setTimelineIndex(newIndex) {
	const max = Math.max(0, forecastTimestamps.length - 1);
	timelineIndex = Math.max(0, Math.min(max, newIndex));
	els.timeSlider.value = String(timelineIndex);
	updateFrame();
}

function getForecastSpeed(detectorId, timestamp) {
	if (!timestamp) {
		return null;
	}

	const speedsForDetector = forecastSpeedsByDetector.get(detectorId);
	if (!speedsForDetector) {
		return null;
	}

	const speed = speedsForDetector.get(timestamp);
	return Number.isFinite(speed) ? speed : null;
}

function updateFrame() {
	const forecastTimestamp = forecastTimestamps[timelineIndex];
	const frameDate = parseForecastTimestamp(forecastTimestamp);
	els.currentPlayingTime.textContent = frameDate ? formatDateTime(frameDate) : "--";

	updateMarkers();
}

function stopPlay(force_stop=false) {
	isPlaying = false;
	if (playTimer) {
		clearInterval(playTimer);
		playTimer = null;
	}
	updatePlayButton();
	if (force_stop == false) {
		startPlay();
	}
}

function startPlay() {
	if (detectorMarkers.length === 0 || forecastTimestamps.length <= 1) {
		return;
	}

	if (timelineIndex >= forecastTimestamps.length - 1) {
		setTimelineIndex(0);
	}

	isPlaying = true;
	updatePlayButton();
	playTimer = setInterval(() => {
		if (timelineIndex >= forecastTimestamps.length - 1) {
			//stopPlay();
			stopPlay();
			return;
		}
		setTimelineIndex(timelineIndex + 1);
	}, PLAY_INTERVAL_MS);
}

function togglePlay() {
	if (isPlaying) {
		stopPlay(true);
	} else {
		startPlay();
	}
}

function updateRealLifeClock() {
	els.realLifeTime.textContent = formatDateTime(new Date());
}

function bindControls() {
	bindMapStyleControl();

	els.timeSlider.min = "0";
	els.timeSlider.max = "0";
	els.timeSlider.value = "0";

	els.btnBackward.addEventListener("click", () => {
		setTimelineIndex(timelineIndex - 1);
	});

	els.btnForward.addEventListener("click", () => {
		setTimelineIndex(timelineIndex + 1);
	});

	els.btnPlay.addEventListener("click", togglePlay);

	els.timeSlider.addEventListener("input", () => {
		setTimelineIndex(Number(els.timeSlider.value));
	});

	updatePlayButton();
	updateRealLifeClock();
	setInterval(updateRealLifeClock, 1000);
	setInterval(update, 100);
}

function update() {
	//updateFrame();
	const topBar = document.getElementById("topBar");
	if (!topBar) {
		return;
	}
	/*
	const currentAnimationTimestamp = forecastTimestamps[timelineIndex];
	const animationDate = parseForecastTimestamp(currentAnimationTimestamp);
	const isFutureTime = animationDate ? animationDate.getTime() > Date.now() : false;

	if (isFutureTime) {
		topBar.classList.add("top-bar-forcast");
	} else {
		topBar.classList.remove("top-bar-forcast");
	}
	*/
}

function updateMarkers(detector=null, marker=null) {
	const forecastTimestamp = forecastTimestamps[timelineIndex];
	const frameDate = parseForecastTimestamp(forecastTimestamp);

	if (detector)
	{
		const speed = getForecastSpeed(detector.detectorId, forecastTimestamp);
		const color = speedToColor(speed);

		marker.setStyle({
			color,
			fillColor: color,
			fillOpacity: 0.8,
		});

		if (marker.isPopupOpen()) {
			marker.setPopupContent(buildDetectorPopup(detector, speed, frameDate));
		}

	}else{

		detectorMarkers.forEach((entry) => {
			const speed = getForecastSpeed(entry.detectorId, forecastTimestamp);
			const color = speedToColor(speed);
			
			entry.marker.setStyle({
				color,
				fillColor: color,
				fillOpacity: 0.8,
			});

			if (entry.marker.isPopupOpen()) {
				entry.marker.setPopupContent(buildDetectorPopup(entry, speed, frameDate));
			}
		});
	}
}

function configureTimeline() {
	if (forecastTimestamps.length === 0) {
		els.labelMinTime.textContent = "--";
		els.labelMaxTime.textContent = "--";
		els.timeSlider.min = "0";
		els.timeSlider.max = "0";
		els.timeSlider.value = "0";
		return;
	}

	els.labelMinTime.textContent = formatForecastLabel(forecastTimestamps[0]);
	els.labelMaxTime.textContent = formatForecastLabel(forecastTimestamps[forecastTimestamps.length - 1]);
	els.timeSlider.min = "0";
	els.timeSlider.max = String(forecastTimestamps.length - 1);
	els.timeSlider.value = String(Math.min(timelineIndex, forecastTimestamps.length - 1));
}

function buildTrafficImageUrl(detectorId) {
	const safeDetectorId = encodeURIComponent((detectorId || "").trim());
	return `https://tdcctv.data.one.gov.hk/${safeDetectorId}.JPG`;
}

function speedTagClass(speed) {
	if (!Number.isFinite(speed)) return "unknown";
	if (speed < 35) return "slow";
	if (speed < 60) return "medium";
	return "fast";
}

function buildDetectorPopup(detector, speed, frameDate) {
	const imageUrl = buildTrafficImageUrl(detector.detectorId);
	const speedClass = speedTagClass(speed);
	const speedText = Number.isFinite(speed) ? `${speed.toFixed(1)} km/h` : "No forecast";
	const timeText = frameDate ? formatDateTime(frameDate) : "--";
	return `<div class="detector-popup"><div class="detector-popup-head"><span class="detector-id">${detector.detectorId}</span><span class="speed-tag ${speedClass}">${speedText}</span></div><div class="detector-popup-body"><div class="detector-road">${detector.roadEn}</div><div class="detector-meta"><span class="label">Direction</span><span class="value">${detector.direction}</span></div><img class="detector-image" src="${imageUrl}" alt="Realtime traffic image for ${detector.detectorId}" loading="lazy" /></div></div>`;
}

function addDetectorsToMap(detectors) {
	const forecastTimestamp = forecastTimestamps[timelineIndex];
	const frameDate = parseForecastTimestamp(forecastTimestamp);

	detectors.forEach((detector) => {
		const speed = getForecastSpeed(detector.detectorId, forecastTimestamp);
		const color = speedToColor(speed);
		const marker = L.circleMarker([detector.lat, detector.lng], {
			renderer,
			radius: 5,
			color,
			weight: 1,
			fillColor: color,
			fillOpacity: 0.8,
		}).addTo(map);

		marker.bindPopup(buildDetectorPopup(detector, speed, frameDate));

		marker.on("click", () => {
			updateMarkers(detector, marker);
		});

		detectorMarkers.push({
			detectorId: detector.detectorId,
			roadEn: detector.roadEn,
			direction: detector.direction,
			marker,
		});
	});
}

function buildForecastIndex(records) {
	const timestamps = [];
	const speedsByDetector = new Map();

	Object.entries(records || {}).forEach(([detectorId, forecast]) => {
		const ids = String(detectorId || "").trim();
		const timestampList = Array.isArray(forecast?.timestamp) ? forecast.timestamp : [];
		const speedList = Array.isArray(forecast?.predicted_speed) ? forecast.predicted_speed : [];

		if (!ids || timestampList.length === 0 || timestampList.length !== speedList.length) {
			return;
		}

		if (!speedsByDetector.has(ids)) {
			speedsByDetector.set(ids, new Map());
		}

		const detectorSpeeds = speedsByDetector.get(ids);

		timestampList.forEach((timestamp, index) => {
			const normalizedTimestamp = String(timestamp || "").trim();
			const speed = Number(speedList[index]);

			if (!normalizedTimestamp || !Number.isFinite(speed)) {
				return;
			}

			if (!timestamps.includes(normalizedTimestamp)) {
				timestamps.push(normalizedTimestamp);
			}

			detectorSpeeds.set(normalizedTimestamp, speed);
		});
	});

	return {
		timestamps: timestamps.sort(),
		speedsByDetector,
	};
}

function normalizeDetectors(rows) {
	return rows
		.map((r) => {
			const lat = Number(r.Latitude);
			const lng = Number(r.Longitude);
			if (!Number.isFinite(lat) || !Number.isFinite(lng)) {
				return null;
			}

			return {
				detectorId: r.AID_ID_Number || "Unknown Detector",
				roadEn: r.Road_EN || "Unknown Road",
				direction: r.Direction || "-",
				lat,
				lng,
			};
		})
		.filter(Boolean);
}

async function loadDetectors() {
	const response = await fetch("rsc/traffic_speed_volume_occ_info_filtered.csv");
	if (!response.ok) {
		throw new Error(`Failed to load detector CSV: HTTP ${response.status}`);
	}

	const csvText = await response.text();
	const rows = parseCsv(csvText);
	const detectors = normalizeDetectors(rows);

	if (detectors.length === 0) {
		throw new Error("No valid detector rows found in CSV.");
	}

	return detectors;
}

async function loadForecastSpeeds() {
	const cacheBuster = `ts=${Date.now()}`;
	const jsonUrl = FORECAST_DATA_URL + "?time=" + Date.now();
	const separator = jsonUrl.includes("?") ? "&" : "?";
	const response = await fetch(`${jsonUrl}${separator}${cacheBuster}`, {
		cache: "no-store",
	});
	if (!response.ok) {
		throw new Error(`Failed to load forecast JSON: HTTP ${response.status}`);
	}

	const records = await response.json();
	if (!records || typeof records !== "object") {
		throw new Error("No forecast records found in JSON.");
	}

	return buildForecastIndex(records);
}

async function refreshForecastSpeeds() {
	if (isRefreshingForecast) {
		return;
	}

	console.log("Refreshing forecast speeds...");

	isRefreshingForecast = true;
	try {
		const forecast = await loadForecastSpeeds();
		if (!forecast.timestamps.length) {
			return;
		}

		forecastTimestamps = forecast.timestamps;
		forecastSpeedsByDetector = forecast.speedsByDetector;
		configureTimeline();
		setTimelineIndex(forecastTimestamps.length - 1);
	} catch (err) {
		console.warn("Forecast refresh failed:", err);
	} finally {
		isRefreshingForecast = false;
	}
}

function startForecastAutoRefresh() {
	if (forecastRefreshTimer) {
		clearInterval(forecastRefreshTimer);
	}

	forecastRefreshTimer = setInterval(refreshForecastSpeeds, FORECAST_REFRESH_INTERVAL_MS);
}

async function init() {
	bindControls();

	try {
		const [detectors, forecast] = await Promise.all([loadDetectors(), loadForecastSpeeds()]);
		forecastTimestamps = forecast.timestamps;
		forecastSpeedsByDetector = forecast.speedsByDetector;
		configureTimeline();
		setTimelineIndex(Math.max(0, forecastTimestamps.length - 1));
		addDetectorsToMap(detectors);
		updateFrame();
		startForecastAutoRefresh();
		startPlay();
	} catch (err) {
		// Display a visible error for local debugging if CSV path or protocol is incorrect.
		const msg = err instanceof Error ? err.message : "Unknown error while loading detectors.";
		alert(`${msg}\n\nTip: open this site via a local web server so fetch() can read CSV files.`);
	}
}

init();
