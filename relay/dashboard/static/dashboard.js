// Dashboard JavaScript

let autoRefreshInterval = null;
let currentFilters = {};

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    loadStats();
    loadJobs();
    setupEventListeners();
    startAutoRefresh();
});

function setupEventListeners() {
    // Refresh button
    document.getElementById('refreshBtn').addEventListener('click', function() {
        loadStats();
        loadJobs();
    });

    // Auto-refresh checkbox
    document.getElementById('autoRefresh').addEventListener('change', function(e) {
        if (e.target.checked) {
            startAutoRefresh();
        } else {
            stopAutoRefresh();
        }
    });

    // Filter inputs
    const filterInputs = [
        'statusFilter', 'providerFilter', 'dateFromFilter', 'dateToFilter',
        'descriptionSearch', 'jobIdSearch'
    ];
    
    filterInputs.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.addEventListener('change', applyFilters);
            element.addEventListener('input', debounce(applyFilters, 300));
        }
    });

    // Clear filters button
    document.getElementById('clearFiltersBtn').addEventListener('click', clearFilters);

    // Modal close
    const modal = document.getElementById('jobModal');
    const closeBtn = document.querySelector('.close');
    closeBtn.addEventListener('click', function() {
        modal.style.display = 'none';
    });
    window.addEventListener('click', function(e) {
        if (e.target === modal) {
            modal.style.display = 'none';
        }
    });
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function startAutoRefresh() {
    stopAutoRefresh();
    autoRefreshInterval = setInterval(() => {
        loadStats();
        loadJobs();
    }, 30000); // 30 seconds
}

function stopAutoRefresh() {
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
        autoRefreshInterval = null;
    }
}

async function loadStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();
        
        if (data.error) {
            console.error('Error loading stats:', data.error);
            return;
        }

        document.getElementById('statTotal').textContent = data.total_jobs || 0;
        document.getElementById('statCompleted').textContent = data.completed || 0;
        document.getElementById('statInProgress').textContent = data.in_progress || 0;
        document.getElementById('statFailed').textContent = data.failed || 0;
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

async function loadJobs() {
    try {
        const params = new URLSearchParams();
        
        if (currentFilters.status) params.append('status', currentFilters.status);
        if (currentFilters.provider) params.append('provider', currentFilters.provider);
        if (currentFilters.date_from) params.append('date_from', currentFilters.date_from);
        if (currentFilters.date_to) params.append('date_to', currentFilters.date_to);
        if (currentFilters.description) params.append('description', currentFilters.description);
        if (currentFilters.job_id) params.append('job_id', currentFilters.job_id);

        const response = await fetch(`/api/jobs?${params.toString()}`);
        const data = await response.json();
        
        if (data.error) {
            document.getElementById('jobsTable').innerHTML = 
                `<div class="error">Error loading jobs: ${data.error}</div>`;
            return;
        }

        renderJobs(data.jobs || []);
    } catch (error) {
        console.error('Error loading jobs:', error);
        document.getElementById('jobsTable').innerHTML = 
            `<div class="error">Error loading jobs: ${error.message}</div>`;
    }
}

function renderJobs(jobs) {
    const tableContainer = document.getElementById('jobsTable');
    document.getElementById('jobCount').textContent = `${jobs.length} job${jobs.length !== 1 ? 's' : ''}`;

    if (jobs.length === 0) {
        tableContainer.innerHTML = '<div class="empty-state">No jobs found</div>';
        return;
    }

    let html = `
        <table>
            <thead>
                <tr>
                    <th>Job ID</th>
                    <th>Provider</th>
                    <th>Status</th>
                    <th>Description</th>
                    <th>Submitted</th>
                    <th>Progress</th>
                    <th>Results</th>
                </tr>
            </thead>
            <tbody>
    `;

    jobs.forEach(job => {
        const submittedDate = new Date(job.submitted_at);
        const dateStr = submittedDate.toLocaleDateString() + ' ' + 
                       submittedDate.toLocaleTimeString();
        
        const normalizedStatus = normalizeStatus(job.status);
        const progress = `${job.completed_requests || 0}/${job.n_requests || 0}`;
        
        html += `
            <tr>
                <td>
                    <a href="#" class="job-id-link" onclick="showJobDetails('${job.job_id}'); return false;">
                        ${escapeHtml(job.job_id)}
                    </a>
                </td>
                <td>${escapeHtml(job.provider || 'N/A')}</td>
                <td>
                    <span class="status-badge status-${normalizedStatus}">
                        ${escapeHtml(job.status)}
                    </span>
                </td>
                <td>${escapeHtml(job.description || 'N/A')}</td>
                <td>${dateStr}</td>
                <td>${progress}</td>
                <td>
                    <span class="results-indicator ${job.has_results ? '' : 'no-results'}" 
                          title="${job.has_results ? 'Results available' : 'No results'}">
                    </span>
                </td>
            </tr>
        `;
    });

    html += `
            </tbody>
        </table>
    `;

    tableContainer.innerHTML = html;
}

function normalizeStatus(status) {
    if (!status) return 'pending';
    const statusLower = status.toLowerCase();
    
    if (['completed', 'ended', 'finalizing'].includes(statusLower)) {
        return 'completed';
    }
    if (['in_progress', 'processing', 'validating'].includes(statusLower)) {
        return 'in_progress';
    }
    if (['failed', 'expired'].includes(statusLower)) {
        return 'failed';
    }
    if (statusLower === 'cancelled') {
        return 'cancelled';
    }
    if (statusLower === 'pending') {
        return 'pending';
    }
    return statusLower;
}

async function showJobDetails(jobId) {
    try {
        const response = await fetch(`/api/jobs/${jobId}`);
        const data = await response.json();
        
        if (data.error) {
            alert('Error loading job details: ' + data.error);
            return;
        }

        const job = data.job;
        const modal = document.getElementById('jobModal');
        const modalContent = document.getElementById('modalContent');
        
        document.getElementById('modalJobId').textContent = `Job: ${job.job_id}`;
        
        const submittedDate = new Date(job.submitted_at);
        const dateStr = submittedDate.toLocaleString();
        
        modalContent.innerHTML = `
            <div class="modal-detail">
                <label>Job ID:</label>
                <span>${escapeHtml(job.job_id)}</span>
            </div>
            <div class="modal-detail">
                <label>Provider:</label>
                <span>${escapeHtml(job.provider || 'N/A')}</span>
            </div>
            <div class="modal-detail">
                <label>Status:</label>
                <span class="status-badge status-${normalizeStatus(job.status)}">
                    ${escapeHtml(job.status)}
                </span>
            </div>
            <div class="modal-detail">
                <label>Description:</label>
                <span>${escapeHtml(job.description || 'N/A')}</span>
            </div>
            <div class="modal-detail">
                <label>Submitted At:</label>
                <span>${dateStr}</span>
            </div>
            <div class="modal-detail">
                <label>Total Requests:</label>
                <span>${job.n_requests || 0}</span>
            </div>
            <div class="modal-detail">
                <label>Completed Requests:</label>
                <span>${job.completed_requests || 0}</span>
            </div>
            <div class="modal-detail">
                <label>Failed Requests:</label>
                <span>${job.failed_requests || 0}</span>
            </div>
            <div class="modal-detail">
                <label>Results Available:</label>
                <span>${job.has_results ? 'Yes' : 'No'}</span>
            </div>
            ${job.provider_job_id ? `
            <div class="modal-detail">
                <label>Provider Job ID:</label>
                <span>${escapeHtml(job.provider_job_id)}</span>
            </div>
            ` : ''}
            <div style="margin-top: 20px;">
                <button class="btn btn-primary" onclick="refreshJob('${job.job_id}')">
                    Refresh Status
                </button>
            </div>
        `;
        
        modal.style.display = 'block';
    } catch (error) {
        alert('Error loading job details: ' + error.message);
    }
}

async function refreshJob(jobId) {
    try {
        const response = await fetch(`/api/jobs/${jobId}/refresh`);
        const data = await response.json();
        
        if (data.error) {
            alert('Error refreshing job: ' + data.error);
            return;
        }

        // Reload stats and jobs
        loadStats();
        loadJobs();
        
        // Update modal if it's open for this job
        if (document.getElementById('modalJobId').textContent.includes(jobId)) {
            showJobDetails(jobId);
        }
    } catch (error) {
        alert('Error refreshing job: ' + error.message);
    }
}

function applyFilters() {
    currentFilters = {
        status: document.getElementById('statusFilter').value,
        provider: document.getElementById('providerFilter').value,
        date_from: document.getElementById('dateFromFilter').value,
        date_to: document.getElementById('dateToFilter').value,
        description: document.getElementById('descriptionSearch').value,
        job_id: document.getElementById('jobIdSearch').value,
    };
    
    loadJobs();
}

function clearFilters() {
    document.getElementById('statusFilter').value = '';
    document.getElementById('providerFilter').value = '';
    document.getElementById('dateFromFilter').value = '';
    document.getElementById('dateToFilter').value = '';
    document.getElementById('descriptionSearch').value = '';
    document.getElementById('jobIdSearch').value = '';
    
    currentFilters = {};
    loadJobs();
}

function escapeHtml(text) {
    if (text === null || text === undefined) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

