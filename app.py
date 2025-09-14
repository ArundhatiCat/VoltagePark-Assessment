# app.py (CogVideoX text->video)
import io, torch, tempfile, os, psutil
import logging
from datetime import datetime
from typing import Dict, List
from enum import Enum
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

# Create logs directory
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# Configure logging
log_file = os.path.join(LOGS_DIR, f"app_{datetime.now().strftime('%Y%m%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Log system startup and GPU information
logger.info("Starting CogVideoX API server")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_count = torch.cuda.device_count()
    cuda_version = torch.version.cuda
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
    
    logger.info(f"GPU Information:")
    logger.info(f"- GPU Name: {gpu_name}")
    logger.info(f"- Number of GPUs: {gpu_count}")
    logger.info(f"- CUDA Version: {cuda_version}")
    logger.info(f"- GPU Memory: {gpu_memory:.2f} GB")
    
    # Log GPU memory usage
    memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
    memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)
    logger.info(f"- Initial GPU Memory Usage:")
    logger.info(f"  - Allocated: {memory_allocated:.2f} GB")
    logger.info(f"  - Reserved: {memory_reserved:.2f} GB")
else:
    logger.warning("No GPU available, using CPU")

# Create output directory if it doesn't exist
OUTPUT_DIR = "output_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)
logger.info(f"Output directory created/verified: {OUTPUT_DIR}")

# Store job status
job_store: Dict[str, Dict] = {}

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

app = FastAPI(
    title="Video Generation API",
    description="""
    Text to Video Generation
    
    Available Endpoints:
        
    """,
    version="1.0.0"
)
logger.info("Loading CogVideoX model...")
pipe = CogVideoXPipeline.from_pretrained("zai-org/CogVideoX-2b")
logger.info("Converting model to bfloat16...")
pipe = pipe.to(device="cuda", dtype=torch.bfloat16)
pipe.enable_model_cpu_offload = False
logger.info("Model loaded and configured successfully")

class GenReq(BaseModel):
    prompt: str
    num_frames: int = 49
    guidance_scale: float = 4.5
    seed: int | None = 42
    fps: int = 12

class JobResponse(BaseModel):
    job_id: str
    status: JobStatus

class SystemHealth(BaseModel):
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    active_jobs: int
    completed_jobs: int

@app.post("/generate", response_model=JobResponse)
async def submit_generation(req: GenReq) -> JobResponse:
    """
    Creates a video based on text description.

    Input:
    ------
    prompt: str
        What you want in the video
    
    num_frames: int, optional
        How many frames to generate 
        (default: 49)
    
    guidance_scale: float, optional
        How closely to follow the prompt 
        (default: 4.5)
    
    seed: int, optional
        For reproducible results 
        (default: 42)
    
    fps: int, optional
        Video playback speed 
        (default: 12)

    Returns:
    -------
    JobResponse
        Contains job_id and status for tracking progress
    """
    # Log the incoming request
    logger.info(f"New request: {req.prompt}")
    
    # Create unique job ID
    job_id: str = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Set up job tracking
    job_store[job_id] = {
        "status": JobStatus.PENDING,
        "request": req.dict(),
        "output_path": None,
        "error": None
    }
    
    try:
        # Start processing
        job_store[job_id]["status"] = JobStatus.PROCESSING
        
        # Set up random seed if provided
        g: torch.Generator = torch.Generator(device="cuda")
        if req.seed is not None:
            g.manual_seed(int(req.seed))
        
        # Generate the video
        out = pipe(
            prompt=str(req.prompt),
            num_frames=int(req.num_frames),
            guidance_scale=float(req.guidance_scale),
            generator=g,
        ).frames[0]

        # Save to file
        video_path: str = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")
        export_to_video(out, video_path, fps=int(req.fps))
        
        # Mark as complete
        job_store[job_id]["status"] = JobStatus.COMPLETED
        job_store[job_id]["output_path"] = video_path
        
    except Exception as e:
        # Handle any errors
        logger.error(f"Failed job {job_id}: {str(e)}", exc_info=True)
        job_store[job_id]["status"] = JobStatus.FAILED
        job_store[job_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))

    return JobResponse(job_id=job_id, status=job_store[job_id]["status"])

@app.get("/status/{job_id}", response_model=JobResponse)
async def check_status(job_id: str) -> JobResponse:
    """
    Check the status of a video generation job.

    Input:
    ------
    job_id: str
        The unique identifier of the job to check

    Returns:
    -------
    JobResponse
        Contains:
        - job_id: The ID of the checked job
        - status: Current status (pending/processing/completed/failed)

    Raises:
    ------
    HTTPException (404)
        When job ID is not found
    """
    # Check if job exists
    if job_id not in job_store:
        logger.warning(f"Status check failed: Job {job_id} not found")
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Return job status
    return JobResponse(
        job_id=job_id, 
        status=job_store[job_id]["status"]
    )

@app.get("/video/{job_id}")
async def get_video(job_id: str) -> Response:
    """
    Retrieve and download the generated video.

    Input:
    ------
    job_id: str
        The unique identifier of the video to retrieve

    Returns:
    -------
    Response
        MP4 video file as a downloadable attachment

    Raises:
    ------
    HTTPException
        404: Job not found or video file missing
        400: Video generation not completed
    """
    # Check if job exists
    if job_id not in job_store:
        logger.warning(f"Video not found: Job {job_id}")
        raise HTTPException(
            status_code=404, 
            detail=f"Job not found"
        )
    
    # Get job details
    job = job_store[job_id]
    
    # Check if video is ready
    if job["status"] != JobStatus.COMPLETED:
        logger.warning(f"Video not ready: {job_id}, status: {job['status']}")
        raise HTTPException(
            status_code=400, 
            detail=f"Video not ready. Status: {job['status']}"
        )
    
    # Verify video path exists
    video_path = job["output_path"]
    if not video_path or not os.path.exists(video_path):
        logger.error(f"Video file missing: {video_path}")
        raise HTTPException(
            status_code=404, 
            detail="Video file not found"
        )
    
    # Read and return video file
    with open(video_path, "rb") as file:
        video_bytes = file.read()
    
    headers = {
        'Content-Disposition': f'attachment; filename="{os.path.basename(video_path)}"'
    }
    
    logger.info(f"Video retrieved: {job_id}")
    return Response(
        content=video_bytes,
        media_type="video/mp4",
        headers=headers
    )

@app.get("/health", response_model=SystemHealth)
async def system_health() -> SystemHealth:
    """
    Get system health metrics.

    Returns:
    -------
    SystemHealth
        Contains:
        - cpu_usage: CPU utilization percentage
        - memory_usage: RAM usage percentage
        - gpu_usage: GPU memory usage percentage
        - active_jobs: Number of currently processing jobs
        - completed_jobs: Number of successfully completed jobs
    """
    # Get CPU and memory stats
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    
    # Calculate GPU usage if available
    gpu_percent = 0
    if torch.cuda.is_available():
        gpu_percent = (
            torch.cuda.memory_allocated(0) / 
            torch.cuda.max_memory_allocated(0) * 100
        )
    
    # Count jobs by status
    active_count = len([
        j for j in job_store.values() 
        if j["status"] == JobStatus.PROCESSING
    ])
    completed_count = len([
        j for j in job_store.values() 
        if j["status"] == JobStatus.COMPLETED
    ])
    
    logger.info(f"Health check - CPU: {cpu_percent}%, MEM: {memory_percent}%, GPU: {gpu_percent:.1f}%")
    
    return SystemHealth(
        cpu_usage=cpu_percent,
        memory_usage=memory_percent,
        gpu_usage=gpu_percent,
        active_jobs=active_count,
        completed_jobs=completed_count
    )

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>Video Generation API</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 40px auto;
                    padding: 0 20px;
                    line-height: 1.6;
                }
                h1 {
                    color: #2c3e50;
                    border-bottom: 2px solid #eee;
                    padding-bottom: 10px;
                }
                .description {
                    color: #34495e;
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 5px;
                    margin: 20px 0;
                }
                a {
                    color: #3498db;
                    text-decoration: none;
                }
                a:hover {
                    text-decoration: underline;
                }
            </style>
        </head>
        <body>
            <h1>Video Generation API</h1>
            <div class="description">
                <p>This service provides an interface to generate videos from text prompts 
                using state-of-the-art models optimized for H100 GPUs. Users can submit 
                generation requests, track processing status, and retrieve results via 
                secure endpoints.</p>
                
                <p>The API is designed for reliability with error handling, monitoring, 
                and containerized deployment. Clear documentation and demo scripts are 
                included to help developers integrate quickly.</p>
            </div>
            <p>Visit <a href="/docs">/docs</a> for the complete API documentation.</p>
        </body>
    </html>
    """
