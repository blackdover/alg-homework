/**
 * 前端应用逻辑
 * 与后端 API 通信，管理界面状态
 */

// 全局变量
let currentJobId = null;
let datasets = [];
let algorithmLayers = {};
let pollInterval = null;
let viewerReady = false;
let pendingMapData = null;

// DOM 元素
const datasetSelect = document.getElementById('dataset-select');
const mmsiInput = document.getElementById('mmsi-input');
const speedSegmentSelect = document.getElementById('speed-segment');
const maxSamplesInput = document.getElementById('max-samples');
const runBtn = document.getElementById('run-btn');
const statusMessage = document.getElementById('status-message');
const resultsTable = document.getElementById('results-table');
const resultsBody = document.getElementById('results-body');
const exportCsvBtn = document.getElementById('export-csv');
const refreshResultsBtn = document.getElementById('refresh-results');
const loadingOverlay = document.getElementById('loading-overlay');
const trajectoryMap = document.getElementById('trajectory-map');
const showOriginalCheckbox = document.getElementById('show-original');
const algorithmLayersDiv = document.getElementById('algorithm-layers');

// API 基础 URL
const API_BASE = 'http://localhost:8000';

// ============================================================================
// 初始化
// ============================================================================

document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

// 监听来自 viewer 的准备消息（viewer_ready）
window.addEventListener('message', (event) => {
    const data = event.data;
    if (!data) return;

    if (data.type === 'viewer_ready') {
        viewerReady = true;
        console.log('viewer ready received');
        if (pendingMapData) {
            trajectoryMap.contentWindow.postMessage(pendingMapData, '*');
            pendingMapData = null;
        }
    }
});

function initializeApp() {
    loadDatasets();
    setupEventListeners();
    generateParameterPanels();
}

// ============================================================================
// 事件监听器
// ============================================================================

function setupEventListeners() {
    // 运行按钮
    runBtn.addEventListener('click', handleRunCompression);

    // 导出按钮
    exportCsvBtn.addEventListener('click', exportResultsToCsv);

    // 刷新结果按钮
    refreshResultsBtn.addEventListener('click', refreshResults);

    // 图层开关
    showOriginalCheckbox.addEventListener('change', updateMapLayers);

    // 算法选择变化时更新参数面板
    document.querySelectorAll('.algorithm-list input[type="checkbox"]').forEach(checkbox => {
        checkbox.addEventListener('change', updateParameterPanels);
    });
}

// ============================================================================
// API 通信
// ============================================================================

async function apiRequest(endpoint, options = {}) {
    const url = `${API_BASE}${endpoint}`;
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
        },
    };

    try {
        const response = await fetch(url, { ...defaultOptions, ...options });
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        return await response.json();
    } catch (error) {
        console.error('API 请求失败:', error);
        throw error;
    }
}

// 加载数据集列表
async function loadDatasets() {
    try {
        showLoading(true);
        const response = await apiRequest('/datasets');
        datasets = response.datasets;
        updateDatasetSelect();
    } catch (error) {
        showStatusMessage('加载数据集失败: ' + error.message, 'error');
    } finally {
        showLoading(false);
    }
}

// 运行压缩任务
async function handleRunCompression() {
    if (!validateInputs()) {
        return;
    }

    const requestData = buildCompressionRequest();

    try {
        showLoading(true);
        runBtn.disabled = true;
        runBtn.textContent = '提交中...';

        const response = await apiRequest('/run', {
            method: 'POST',
            body: JSON.stringify(requestData)
        });

        currentJobId = response.job_id;
        showStatusMessage('任务已提交，开始执行...', 'info');

        // 开始轮询任务状态
        startPollingJobStatus();

    } catch (error) {
        showStatusMessage('提交任务失败: ' + error.message, 'error');
        runBtn.disabled = false;
        runBtn.textContent = '运行压缩';
    } finally {
        showLoading(false);
    }
}

// 轮询任务状态
function startPollingJobStatus() {
    if (pollInterval) {
        clearInterval(pollInterval);
    }

    pollInterval = setInterval(async () => {
        try {
            const response = await apiRequest(`/jobs/${currentJobId}`);

            if (response.status === 'completed') {
                clearInterval(pollInterval);
                showStatusMessage('压缩完成！', 'success');
                runBtn.disabled = false;
                runBtn.textContent = '运行压缩';
                loadCompressionResults();
            } else if (response.status === 'failed') {
                clearInterval(pollInterval);
                showStatusMessage('压缩失败: ' + response.message, 'error');
                runBtn.disabled = false;
                runBtn.textContent = '运行压缩';
            } else {
                showStatusMessage(`${response.message} (${response.progress.toFixed(1)}%)`, 'info');
            }
        } catch (error) {
            console.error('轮询任务状态失败:', error);
        }
    }, 2000); // 每2秒轮询一次
}

// 加载压缩结果
async function loadCompressionResults() {
    try {
        // 获取指标数据
        const metricsResponse = await apiRequest(`/metrics/${currentJobId}`);

        // 获取完整结果（包含轨迹数据）
        const resultsResponse = await apiRequest(`/results/${currentJobId}`);

        displayResults(metricsResponse.metrics, resultsResponse.results);
        updateMapWithResults(resultsResponse.results);

    } catch (error) {
        showStatusMessage('加载结果失败: ' + error.message, 'error');
    }
}

// ============================================================================
// 界面更新
// ============================================================================

function updateDatasetSelect() {
    datasetSelect.innerHTML = '<option value="">请选择数据集...</option>';

    datasets.forEach(dataset => {
        const option = document.createElement('option');
        option.value = dataset.name;
        option.textContent = `${dataset.name} (${dataset.ship_type || 'Unknown'}, ${dataset.sample_size || 'N/A'} 点)`;
        datasetSelect.appendChild(option);
    });
}

function generateParameterPanels() {
    const container = document.getElementById('params-container');
    container.innerHTML = '';

    const algorithms = [
        { id: 'dr', name: '自适应 DR', params: [
            { key: 'min_threshold', label: '最低阈值 (m)', default: 20 },
            { key: 'max_threshold', label: '最高阈值 (m)', default: 500 },
            { key: 'v_lower', label: '低速截止 (节)', default: 3 },
            { key: 'v_upper', label: '高速截止 (节)', default: 20 }
        ]},
        { id: 'fixed_dr', name: '固定阈值 DR', params: [
            { key: 'epsilon', label: '距离阈值 (m)', default: 100 }
        ]},
        { id: 'semantic_dr', name: '语义增强 DR', params: [
            { key: 'min_threshold', label: '最低阈值 (m)', default: 20 },
            { key: 'max_threshold', label: '最高阈值 (m)', default: 500 },
            { key: 'v_lower', label: '低速截止 (节)', default: 3 },
            { key: 'v_upper', label: '高速截止 (节)', default: 20 },
            { key: 'cog_threshold', label: '转向阈值 (°)', default: 10 },
            { key: 'sog_threshold', label: '速度变化阈值 (节)', default: 1 },
            { key: 'time_threshold', label: '时间间隔阈值 (s)', default: 300 }
        ]},
        { id: 'sliding', name: 'Sliding Window', params: [
            { key: 'epsilon', label: '距离阈值 (m)', default: 100 }
        ]},
        { id: 'opening', name: 'Opening Window', params: [
            { key: 'epsilon', label: '距离阈值 (m)', default: 100 }
        ]},
        { id: 'squish', name: 'SQUISH', params: [
            { key: 'buffer_size', label: '缓冲区大小', default: 100 }
        ]}
    ];

    algorithms.forEach(algo => {
        const panel = document.createElement('div');
        panel.className = 'param-group';
        panel.id = `params-${algo.id}`;
        panel.style.display = 'none'; // 默认隐藏

        panel.innerHTML = `<h5>${algo.name} 参数</h5>`;

        algo.params.forEach(param => {
            const paramRow = document.createElement('div');
            paramRow.className = 'param-row';

            paramRow.innerHTML = `
                <label for="${algo.id}-${param.key}">${param.label}:</label>
                <input type="number" id="${algo.id}-${param.key}"
                       value="${param.default}" step="0.1" min="0">
            `;

            panel.appendChild(paramRow);
        });

        container.appendChild(panel);
    });
}

function updateParameterPanels() {
    const selectedAlgorithms = getSelectedAlgorithms();

    // 显示/隐藏参数面板
    document.querySelectorAll('.param-group').forEach(panel => {
        const algoId = panel.id.replace('params-', '');
        panel.style.display = selectedAlgorithms.includes(algoId) ? 'block' : 'none';
    });
}

function displayResults(metrics, fullResults) {
    resultsBody.innerHTML = '';

    if (!metrics || metrics.length === 0) {
        resultsBody.innerHTML = '<tr><td colspan="7" class="no-data">暂无数据</td></tr>';
        return;
    }

    metrics.forEach(metric => {
        const row = document.createElement('tr');

        const algorithmNames = {
            'dr': '自适应 DR',
            'fixed_dr': '固定阈值 DR',
            'semantic_dr': '语义增强 DR',
            'sliding': 'Sliding Window',
            'opening': 'Opening Window',
            'squish': 'SQUISH'
        };

        row.innerHTML = `
            <td>${algorithmNames[metric.algorithm] || metric.algorithm}</td>
            <td>${metric.compression_ratio ? metric.compression_ratio.toFixed(2) : 'N/A'}</td>
            <td>${metric.elapsed_time ? (metric.elapsed_time * 1000).toFixed(0) : 'N/A'}</td>
            <td>${metric.sed_mean ? metric.sed_mean.toFixed(2) : 'N/A'}</td>
            <td>${metric.sed_max ? metric.sed_max.toFixed(2) : 'N/A'}</td>
            <td>${metric.sed_p95 ? metric.sed_p95.toFixed(2) : 'N/A'}</td>
            <td>${metric.event_recall ? (metric.event_recall * 100).toFixed(2) : 'N/A'}</td>
        `;

        resultsBody.appendChild(row);
    });

    // 生成算法图层开关
    generateAlgorithmLayerSwitches(metrics);
}

function generateAlgorithmLayerSwitches(metrics) {
    algorithmLayersDiv.innerHTML = '';

    const algorithmNames = {
        'dr': '自适应 DR',
        'fixed_dr': '固定阈值 DR',
        'semantic_dr': '语义增强 DR',
        'sliding': 'Sliding Window',
        'opening': 'Opening Window',
        'squish': 'SQUISH'
    };

    metrics.forEach(metric => {
        // 使用 <label> 包裹保证点击任意区域都能切换 checkbox
        const layerLabel = document.createElement('label');
        layerLabel.className = 'switch-label';

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `show-${metric.algorithm}`;
        checkbox.checked = true; // 默认显示
        checkbox.style.marginRight = '8px';

        const slider = document.createElement('span');
        slider.className = 'switch-slider';
        slider.style.marginRight = '8px';

        const textNode = document.createTextNode(` ${algorithmNames[metric.algorithm] || metric.algorithm}`);

        layerLabel.appendChild(checkbox);
        layerLabel.appendChild(slider);
        layerLabel.appendChild(textNode);

        // 点击 label 任意位置都会触发 checkbox 的 change
        checkbox.addEventListener('change', updateMapLayers);

        algorithmLayersDiv.appendChild(layerLabel);
        algorithmLayers[metric.algorithm] = checkbox;
    });
}

function updateMapWithResults(results) {
    // 通过 postMessage 向地图 iframe 发送数据
    const mapData = {
        type: 'update_trajectories',
        original: null, // 如果有原始轨迹数据可以添加
        compressed: {}
    };

    results.forEach(result => {
        mapData.compressed[result.algorithm] = result.compressed_trajectory.map(point => ({
            lat: point.lat,
            lon: point.lon,
            timestamp: point.timestamp
        }));
    });
    // 如果 viewer 尚未就绪，缓存数据并等候 ready 消息
    if (!viewerReady) {
        pendingMapData = mapData;
        console.log('viewer not ready, caching map data');
        return;
    }

    trajectoryMap.contentWindow.postMessage(mapData, '*');
}

function updateMapLayers() {
    const layerStates = {
        original: showOriginalCheckbox.checked,
        algorithms: {}
    };

    Object.keys(algorithmLayers).forEach(algo => {
        layerStates.algorithms[algo] = algorithmLayers[algo].checked;
    });

    trajectoryMap.contentWindow.postMessage({
        type: 'update_layers',
        layers: layerStates
    }, '*');
}

function exportResultsToCsv() {
    const rows = [];
    const headers = ['算法', '压缩率 (%)', '运行时间 (ms)', 'SED Mean (m)', 'SED Max (m)', 'SED P95 (m)', '事件保留率 (%)'];

    rows.push(headers.join(','));

    const tableRows = resultsBody.querySelectorAll('tr');
    tableRows.forEach(row => {
        const cells = row.querySelectorAll('td');
        if (cells.length > 0) {
            const rowData = Array.from(cells).map(cell => cell.textContent);
            rows.push(rowData.join(','));
        }
    });

    const csvContent = rows.join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');

    if (link.download !== undefined) {
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', 'compression_results.csv');
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
}

function refreshResults() {
    if (currentJobId) {
        loadCompressionResults();
    } else {
        showStatusMessage('没有可刷新的结果', 'error');
    }
}

// ============================================================================
// 工具函数
// ============================================================================

function getSelectedAlgorithms() {
    const checkboxes = document.querySelectorAll('.algorithm-list input[type="checkbox"]:checked');
    return Array.from(checkboxes).map(cb => cb.value);
}

function buildCompressionRequest() {
    const selectedAlgorithms = getSelectedAlgorithms();

    // 构建算法参数
    const params = {};
    selectedAlgorithms.forEach(algo => {
        params[algo] = {};

        // 收集该算法的参数
        const paramInputs = document.querySelectorAll(`#params-${algo} input`);
        paramInputs.forEach(input => {
            const paramKey = input.id.replace(`${algo}-`, '');
            const value = parseFloat(input.value);
            if (!isNaN(value)) {
                params[algo][paramKey] = value;
            }
        });
    });

    return {
        dataset_name: datasetSelect.value,
        algorithms: selectedAlgorithms,
        params: params,
        mmsi: mmsiInput.value ? parseInt(mmsiInput.value) : null,
        speed_segment: speedSegmentSelect.value || null,
        max_samples: maxSamplesInput.value ? parseInt(maxSamplesInput.value) : null
    };
}

function validateInputs() {
    if (!datasetSelect.value) {
        showStatusMessage('请选择数据集', 'error');
        return false;
    }

    const selectedAlgorithms = getSelectedAlgorithms();
    if (selectedAlgorithms.length === 0) {
        showStatusMessage('请至少选择一个算法', 'error');
        return false;
    }

    return true;
}

function showStatusMessage(message, type = 'info') {
    statusMessage.textContent = message;
    statusMessage.className = `status-message ${type}`;
    statusMessage.style.display = 'block';

    // 自动隐藏成功和错误消息
    if (type === 'success' || type === 'error') {
        setTimeout(() => {
            statusMessage.style.display = 'none';
        }, 5000);
    }
}

function showLoading(show) {
    loadingOverlay.style.display = show ? 'flex' : 'none';
}
