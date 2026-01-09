"""Web dashboard for monitoring Relay batch jobs."""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from flask import Flask, jsonify, request, send_from_directory
from relay.client import RelayClient


def create_app(workspace_dir: str) -> Flask:
    """Create and configure the Flask application.
    
    Args:
        workspace_dir: Path to the workspace directory
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__, static_folder=None)
    
    # Initialize RelayClient
    client = RelayClient(directory=workspace_dir)
    
    # Get the directory where static files are located
    dashboard_dir = Path(__file__).parent / "dashboard" / "static"
    
    @app.route("/")
    def index():
        """Serve the main dashboard page."""
        return send_from_directory(dashboard_dir, "index.html")
    
    @app.route("/<path:filename>")
    def static_files(filename: str):
        """Serve static files (CSS, JS)."""
        return send_from_directory(dashboard_dir, filename)
    
    @app.route("/api/jobs", methods=["GET"])
    def get_jobs():
        """Get all jobs with optional filtering.
        
        Query parameters:
            status: Filter by normalized status
            provider: Filter by provider name
            date_from: Filter jobs submitted after this date (ISO format)
            date_to: Filter jobs submitted before this date (ISO format)
            description: Search description text
            job_id: Search job ID
        """
        try:
            status = request.args.get("status")
            provider = request.args.get("provider")
            date_from = request.args.get("date_from")
            date_to = request.args.get("date_to")
            description_search = request.args.get("description")
            job_id_search = request.args.get("job_id")
            
            jobs = client.get_all_jobs(
                status=status,
                provider=provider,
                date_from=date_from,
                date_to=date_to,
                description_search=description_search,
                job_id_search=job_id_search,
            )
            
            return jsonify({"jobs": jobs})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/jobs/<job_id>", methods=["GET"])
    def get_job(job_id: str):
        """Get detailed information about a specific job.
        
        Args:
            job_id: The job ID to retrieve
        """
        try:
            job = client.get_job(job_id)
            if job is None:
                return jsonify({"error": f"Job {job_id} not found"}), 404
            
            # Add has_results field
            job["has_results"] = client.has_results(job_id)
            
            return jsonify({"job": job})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/jobs/<job_id>/refresh", methods=["GET"])
    def refresh_job(job_id: str):
        """Refresh job status from provider API.
        
        Args:
            job_id: The job ID to refresh
        """
        try:
            job_status = client.monitor_batch(job_id)
            
            # Get updated job metadata
            job = client.get_job(job_id)
            if job:
                job["has_results"] = client.has_results(job_id)
                return jsonify({"job": job})
            else:
                return jsonify({"error": f"Job {job_id} not found"}), 404
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/stats", methods=["GET"])
    def get_stats():
        """Get dashboard statistics."""
        try:
            all_jobs = client.get_all_jobs()
            
            total_jobs = len(all_jobs)
            completed = 0
            in_progress = 0
            failed = 0
            cancelled = 0
            pending = 0
            
            for job in all_jobs:
                normalized_status = client._normalize_status(job.get("status", ""))
                if normalized_status == "completed":
                    completed += 1
                elif normalized_status == "in_progress":
                    in_progress += 1
                elif normalized_status == "failed":
                    failed += 1
                elif normalized_status == "cancelled":
                    cancelled += 1
                elif normalized_status == "pending":
                    pending += 1
            
            return jsonify({
                "total_jobs": total_jobs,
                "completed": completed,
                "in_progress": in_progress,
                "failed": failed,
                "cancelled": cancelled,
                "pending": pending,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return app


def run_dashboard(workspace_dir: str, host: str = "127.0.0.1", port: int = 5000, debug: bool = False):
    """Run the dashboard web server.
    
    Args:
        workspace_dir: Path to the workspace directory
        host: Host to bind to (default: 127.0.0.1)
        port: Port to bind to (default: 5000)
        debug: Enable debug mode (default: False)
    """
    app = create_app(workspace_dir)
    print(f"Starting Relay dashboard on http://{host}:{port}")
    print(f"Workspace directory: {workspace_dir}")
    app.run(host=host, port=port, debug=debug)


def main():
    """CLI entry point for the dashboard."""
    parser = argparse.ArgumentParser(description="Launch Relay dashboard")
    parser.add_argument(
        "workspace_dir",
        help="Path to workspace directory"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to bind to (default: 5000)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Validate workspace directory
    workspace_path = Path(args.workspace_dir)
    if not workspace_path.exists():
        print(f"Error: Workspace directory '{args.workspace_dir}' does not exist")
        sys.exit(1)
    
    run_dashboard(
        workspace_dir=str(workspace_path.resolve()),
        host=args.host,
        port=args.port,
        debug=args.debug
    )

