import asyncio
import io
import logging
import mimetypes
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import aiofiles
import chardet
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    magic = None
import nltk
import tiktoken
from bs4 import BeautifulSoup
from docx import Document as DocxDocument

# PDF processing libraries (multiple options for robustness)
try:
    from PyPDF2 import PdfReader as PyPDF2Reader
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    from pypdf import PdfReader as PyPDFReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

from ..core.config import settings
from ..models.document_models import DocumentChunk, ProcessingResult
from .openai_service import OpenAIService

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Comprehensive document processing service that handles:
    - Multiple file format support (PDF, DOCX, TXT, HTML, etc.)
    - Intelligent text extraction
    - Smart chunking strategies
    - Metadata extraction
    """
    
    def __init__(self, openai_service: OpenAIService):
        self.openai_service = openai_service
        self.encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        self._ensure_nltk_data()
        self._check_pdf_libraries()
    
    def _ensure_nltk_data(self):
        """Ensure required NLTK data is downloaded"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
    
    def _check_pdf_libraries(self):
        """Check and log available PDF processing libraries"""
        available_libs = []
        if PYMUPDF_AVAILABLE:
            available_libs.append("PyMuPDF")
        if PDFPLUMBER_AVAILABLE:
            available_libs.append("pdfplumber")
        if PYPDF_AVAILABLE:
            available_libs.append("pypdf")
        if PYPDF2_AVAILABLE:
            available_libs.append("PyPDF2")
        
        if not available_libs:
            logger.error("No PDF processing libraries available! Install at least one: pymupdf, pdfplumber, pypdf, or PyPDF2")
        else:
            logger.info(f"Available PDF processing libraries: {', '.join(available_libs)}")
            if not PYMUPDF_AVAILABLE:
                logger.warning("PyMuPDF not available - consider installing it for better PDF processing: pip install pymupdf")
    
    async def process_document(
        self,
        file_content: bytes,
        filename: str,
        content_type: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        metadata: Optional[Dict] = None
    ) -> ProcessingResult:
        """
        Process a document and return chunks with embeddings
        
        Args:
            file_content: Raw file bytes
            filename: Original filename
            content_type: MIME type (auto-detected if None)
            chunk_size: Target size for text chunks (in tokens)
            chunk_overlap: Overlap between chunks (in tokens)
            metadata: Additional metadata to store
            
        Returns:
            ProcessingResult with extracted text, chunks, and metadata
        """
        try:
            logger.info(f"Processing document: {filename}")
            
            # Detect content type if not provided
            if not content_type:
                content_type = self._detect_content_type(file_content, filename)
            
            # Extract text based on file type
            extracted_text = await self._extract_text(file_content, content_type, filename)
            
            if not extracted_text.strip():
                raise ValueError("No text content could be extracted from the document")
            
            # Generate chunks
            chunks = await self._create_chunks(
                text=extracted_text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                filename=filename
            )
            
            # Generate embeddings for chunks
            chunks_with_embeddings = await self._generate_embeddings(chunks)
            
            # Extract document metadata
            doc_metadata = self._extract_metadata(file_content, content_type, filename)
            if metadata:
                doc_metadata.update(metadata)
            
            result = ProcessingResult(
                filename=filename,
                content_type=content_type,
                text_content=extracted_text,
                chunks=chunks_with_embeddings,
                metadata=doc_metadata,
                processing_stats={
                    "total_chunks": len(chunks_with_embeddings),
                    "total_tokens": len(self.encoding.encode(extracted_text)),
                    "avg_chunk_size": sum(len(self.encoding.encode(chunk.text)) for chunk in chunks_with_embeddings) / len(chunks_with_embeddings) if chunks_with_embeddings else 0
                }
            )
            
            logger.info(f"Successfully processed {filename}: {len(chunks_with_embeddings)} chunks generated")
            return result
            
        except Exception as e:
            logger.error(f"Error processing document {filename}: {str(e)}")
            raise
    
    def _detect_content_type(self, file_content: bytes, filename: str) -> str:
        """Detect MIME type of file content"""
        if MAGIC_AVAILABLE:
            try:
                # Try python-magic first (more accurate)
                mime_type = magic.from_buffer(file_content, mime=True)
                if mime_type and mime_type != 'application/octet-stream':
                    return mime_type
            except Exception as e:
                logger.warning(f"Magic library failed: {e}")
        
        # Fallback to mimetypes based on filename
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type or 'application/octet-stream'
    
    async def _extract_text(self, file_content: bytes, content_type: str, filename: str) -> str:
        """Extract text content based on file type"""
        try:
            if content_type.startswith('text/'):
                return await self._extract_text_plain(file_content)
            elif content_type == 'application/pdf':
                return await self._extract_text_pdf(file_content)
            elif content_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/msword']:
                return await self._extract_text_docx(file_content)
            elif content_type in ['text/html', 'application/xhtml+xml']:
                return await self._extract_text_html(file_content)
            else:
                # Try as plain text as fallback
                logger.warning(f"Unknown content type {content_type} for {filename}, trying as plain text")
                return await self._extract_text_plain(file_content)
        except Exception as e:
            logger.error(f"Error extracting text from {filename}: {str(e)}")
            raise
    
    async def _extract_text_plain(self, file_content: bytes) -> str:
        """Extract text from plain text files"""
        # Detect encoding
        detected = chardet.detect(file_content)
        encoding = detected.get('encoding', 'utf-8')
        
        try:
            return file_content.decode(encoding)
        except UnicodeDecodeError:
            # Fallback to utf-8 with error handling
            return file_content.decode('utf-8', errors='replace')
    
    async def _extract_text_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF files using multiple fallback methods"""
        errors = []
        
        # Method 1: Try PyMuPDF (fitz) - often most reliable
        if PYMUPDF_AVAILABLE:
            try:
                return await self._extract_pdf_with_pymupdf(file_content)
            except Exception as e:
                errors.append(f"PyMuPDF failed: {str(e)}")
                logger.warning(f"PyMuPDF extraction failed: {str(e)}")
        
        # Method 2: Try pdfplumber - good for complex layouts
        if PDFPLUMBER_AVAILABLE:
            try:
                return await self._extract_pdf_with_pdfplumber(file_content)
            except Exception as e:
                errors.append(f"pdfplumber failed: {str(e)}")
                logger.warning(f"pdfplumber extraction failed: {str(e)}")
        
        # Method 3: Try pypdf (newer version of PyPDF2)
        if PYPDF_AVAILABLE:
            try:
                return await self._extract_pdf_with_pypdf(file_content)
            except Exception as e:
                errors.append(f"pypdf failed: {str(e)}")
                logger.warning(f"pypdf extraction failed: {str(e)}")
        
        # Method 4: Try PyPDF2 as last resort
        if PYPDF2_AVAILABLE:
            try:
                return await self._extract_pdf_with_pypdf2(file_content)
            except Exception as e:
                errors.append(f"PyPDF2 failed: {str(e)}")
                logger.warning(f"PyPDF2 extraction failed: {str(e)}")
        
        # If all methods failed, raise comprehensive error
        error_msg = f"All PDF extraction methods failed. Errors: {'; '.join(errors)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    async def _extract_pdf_with_pymupdf(self, file_content: bytes) -> str:
        """Extract text using PyMuPDF (fitz)"""
        pdf_document = fitz.open(stream=file_content, filetype="pdf")
        text_parts = []
        
        for page_num in range(pdf_document.page_count):
            try:
                page = pdf_document[page_num]
                page_text = page.get_text()
                if page_text.strip():
                    text_parts.append(page_text)
            except Exception as e:
                logger.warning(f"Error extracting text from page {page_num} with PyMuPDF: {str(e)}")
                continue
        
        pdf_document.close()
        return '\n\n'.join(text_parts)
    
    async def _extract_pdf_with_pdfplumber(self, file_content: bytes) -> str:
        """Extract text using pdfplumber"""
        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            text_parts = []
            
            for page_num, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_parts.append(page_text)
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num} with pdfplumber: {str(e)}")
                    continue
            
            return '\n\n'.join(text_parts)
    
    async def _extract_pdf_with_pypdf(self, file_content: bytes) -> str:
        """Extract text using pypdf"""
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDFReader(pdf_file)
        
        text_parts = []
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text.strip():
                    text_parts.append(page_text)
            except Exception as e:
                logger.warning(f"Error extracting text from page {page_num} with pypdf: {str(e)}")
                continue
        
        return '\n\n'.join(text_parts)
    
    async def _extract_pdf_with_pypdf2(self, file_content: bytes) -> str:
        """Extract text using PyPDF2 (fallback method)"""
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2Reader(pdf_file)
        
        text_parts = []
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text.strip():
                    text_parts.append(page_text)
            except Exception as e:
                logger.warning(f"Error extracting text from page {page_num} with PyPDF2: {str(e)}")
                continue
        
        return '\n\n'.join(text_parts)
    
    async def _extract_text_docx(self, file_content: bytes) -> str:
        """Extract text from DOCX files"""
        try:
            docx_file = io.BytesIO(file_content)
            doc = DocxDocument(docx_file)
            
            text_parts = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text)
            
            return '\n\n'.join(text_parts)
        except Exception as e:
            logger.error(f"Error processing DOCX: {str(e)}")
            raise
    
    async def _extract_text_html(self, file_content: bytes) -> str:
        """Extract text from HTML files"""
        try:
            # Detect encoding
            detected = chardet.detect(file_content)
            encoding = detected.get('encoding', 'utf-8')
            
            html_content = file_content.decode(encoding, errors='replace')
            soup = BeautifulSoup(html_content, 'lxml')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text and clean it up
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            logger.error(f"Error processing HTML: {str(e)}")
            raise
    
    async def _create_chunks(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        filename: str
    ) -> List[DocumentChunk]:
        """Create intelligent text chunks"""
        try:
            # Clean and normalize text
            cleaned_text = self._clean_text(text)
            
            # Use sentence-aware chunking
            chunks = await self._sentence_aware_chunking(
                cleaned_text, chunk_size, chunk_overlap
            )
            
            # Create DocumentChunk objects
            document_chunks = []
            for i, chunk_text in enumerate(chunks):
                chunk = DocumentChunk(
                    text=chunk_text,
                    chunk_index=i,
                    token_count=len(self.encoding.encode(chunk_text)),
                    metadata={
                        "source_file": filename,
                        "chunk_method": "sentence_aware",
                        "chunk_size_target": chunk_size,
                        "chunk_overlap": chunk_overlap
                    }
                )
                document_chunks.append(chunk)
            
            return document_chunks
            
        except Exception as e:
            logger.error(f"Error creating chunks for {filename}: {str(e)}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive newlines but preserve paragraph breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove special characters that might interfere with processing
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', '', text)
        
        return text.strip()
    
    async def _sentence_aware_chunking(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[str]:
        """Create chunks that respect sentence boundaries"""
        try:
            # Split into sentences
            sentences = nltk.sent_tokenize(text)
            
            chunks = []
            current_chunk = []
            current_tokens = 0
            
            for sentence in sentences:
                sentence_tokens = len(self.encoding.encode(sentence))
                
                # If adding this sentence would exceed chunk size, finalize current chunk
                if current_tokens + sentence_tokens > chunk_size and current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(chunk_text)
                    
                    # Start new chunk with overlap
                    overlap_text = self._create_overlap(chunk_text, chunk_overlap)
                    if overlap_text:
                        current_chunk = [overlap_text]
                        current_tokens = len(self.encoding.encode(overlap_text))
                    else:
                        current_chunk = []
                        current_tokens = 0
                
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
            
            # Add final chunk if it has content
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            return chunks
            
        except Exception as e:
            logger.warning(f"Sentence-aware chunking failed, falling back to simple chunking: {str(e)}")
            return await self._simple_chunking(text, chunk_size, chunk_overlap)
    
    def _create_overlap(self, text: str, overlap_tokens: int) -> str:
        """Create overlap text from the end of a chunk"""
        if overlap_tokens <= 0:
            return ""
        
        tokens = self.encoding.encode(text)
        if len(tokens) <= overlap_tokens:
            return text
        
        overlap_token_slice = tokens[-overlap_tokens:]
        return self.encoding.decode(overlap_token_slice)
    
    async def _simple_chunking(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Simple token-based chunking as fallback"""
        tokens = self.encoding.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # Move start position with overlap
            start = end - chunk_overlap if chunk_overlap > 0 else end
        
        return chunks
    
    async def _generate_embeddings(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Generate embeddings for document chunks"""
        try:
            # Extract text from chunks
            texts = [chunk.text for chunk in chunks]
            
            # Generate embeddings in batches
            embeddings = await self.openai_service.create_embeddings_batch(texts)
            
            # Add embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def _extract_metadata(self, file_content: bytes, content_type: str, filename: str) -> Dict:
        """Extract metadata from document"""
        metadata = {
            "filename": filename,
            "content_type": content_type,
            "file_size": len(file_content),
            "processing_timestamp": None  # Will be set by the caller
        }
        
        # Add file extension
        file_path = Path(filename)
        metadata["file_extension"] = file_path.suffix.lower()
        
        # Add content-type specific metadata
        if content_type == 'application/pdf':
            try:
                pdf_file = io.BytesIO(file_content)
                pdf_reader = PdfReader(pdf_file)
                metadata.update({
                    "page_count": len(pdf_reader.pages),
                    "pdf_metadata": pdf_reader.metadata._data if pdf_reader.metadata else {}
                })
            except Exception:
                pass
        
        return metadata
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        return [
            'text/plain',
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/msword',
            'text/html',
            'application/xhtml+xml',
            'text/markdown',
            'text/csv'
        ]
    
    async def validate_file(self, file_content: bytes, filename: str, max_size_mb: int = 50) -> Tuple[bool, str]:
        """Validate if file can be processed"""
        # Check file size
        size_mb = len(file_content) / (1024 * 1024)
        if size_mb > max_size_mb:
            return False, f"File size ({size_mb:.1f}MB) exceeds maximum allowed size ({max_size_mb}MB)"
        
        # Check content type
        content_type = self._detect_content_type(file_content, filename)
        supported_formats = self.get_supported_formats()
        
        if content_type not in supported_formats:
            return False, f"File type '{content_type}' is not supported. Supported formats: {', '.join(supported_formats)}"
        
        # For PDF files, do a more thorough validation since they can be tricky
        if content_type == 'application/pdf':
            return await self._validate_pdf_file(file_content, filename)
        
        # For non-PDF files, try to extract a small sample to verify the file is readable
        try:
            # Use a smaller sample for validation to avoid processing large files
            sample_size = min(5000, len(file_content))
            sample_text = await self._extract_text(file_content[:sample_size], content_type, filename)
            if not sample_text.strip():
                return False, "File appears to be empty or contains no extractable text"
        except Exception as e:
            return False, f"File validation failed: {str(e)}"
        
        return True, "File is valid and can be processed"
    
    async def _validate_pdf_file(self, file_content: bytes, filename: str) -> Tuple[bool, str]:
        """Specialized validation for PDF files"""
        validation_errors = []
        
        # Check if any PDF library can at least open the file
        pdf_openable = False
        
        # Try PyMuPDF first (most robust)
        if PYMUPDF_AVAILABLE:
            try:
                pdf_doc = fitz.open(stream=file_content, filetype="pdf")
                if pdf_doc.page_count > 0:
                    pdf_openable = True
                    # Try to extract text from first page
                    try:
                        first_page = pdf_doc[0]
                        sample_text = first_page.get_text()
                        pdf_doc.close()
                        if sample_text.strip():
                            return True, "PDF file is valid and contains extractable text"
                        else:
                            logger.warning(f"PDF {filename} opened but contains no extractable text")
                    except Exception:
                        pdf_doc.close()
                        pass
                else:
                    pdf_doc.close()
            except Exception as e:
                validation_errors.append(f"PyMuPDF: {str(e)}")
        
        # Try pdfplumber
        if PDFPLUMBER_AVAILABLE and not pdf_openable:
            try:
                with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                    if len(pdf.pages) > 0:
                        pdf_openable = True
                        # Try to extract text from first page
                        try:
                            sample_text = pdf.pages[0].extract_text()
                            if sample_text and sample_text.strip():
                                return True, "PDF file is valid and contains extractable text"
                        except Exception:
                            pass
            except Exception as e:
                validation_errors.append(f"pdfplumber: {str(e)}")
        
        # Try pypdf
        if PYPDF_AVAILABLE and not pdf_openable:
            try:
                pdf_reader = PyPDFReader(io.BytesIO(file_content))
                if len(pdf_reader.pages) > 0:
                    pdf_openable = True
                    # Try to extract text from first page
                    try:
                        sample_text = pdf_reader.pages[0].extract_text()
                        if sample_text.strip():
                            return True, "PDF file is valid and contains extractable text"
                    except Exception:
                        pass
            except Exception as e:
                validation_errors.append(f"pypdf: {str(e)}")
        
        # Try PyPDF2 as last resort
        if PYPDF2_AVAILABLE and not pdf_openable:
            try:
                pdf_reader = PyPDF2Reader(io.BytesIO(file_content))
                if len(pdf_reader.pages) > 0:
                    pdf_openable = True
                    # Try to extract text from first page
                    try:
                        sample_text = pdf_reader.pages[0].extract_text()
                        if sample_text.strip():
                            return True, "PDF file is valid and contains extractable text"
                    except Exception:
                        pass
            except Exception as e:
                validation_errors.append(f"PyPDF2: {str(e)}")
        
        if pdf_openable:
            # PDF can be opened but might not have extractable text (e.g., scanned images)
            logger.warning(f"PDF {filename} can be opened but may not contain extractable text")
            return True, "PDF file is valid but may contain limited extractable text (possibly scanned images)"
        
        # PDF cannot be opened by any library
        error_summary = "; ".join(validation_errors) if validation_errors else "Unknown PDF format error"
        return False, f"PDF file appears to be corrupted or in an unsupported format. Errors: {error_summary}" 