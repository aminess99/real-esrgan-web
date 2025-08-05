#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import base64
import subprocess
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import mimetypes
import tempfile
import uuid
from datetime import datetime

class RealESRGANHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.base_path = Path(__file__).parent
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """معالجة طلبات GET"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path == '/' or path == '/index.html':
            self.serve_file('web_interface.html')
        elif path.startswith('/results/'):
            # تقديم الصور المحسنة
            file_path = self.base_path / path[1:]  # إزالة الشرطة المائلة الأولى
            if file_path.exists():
                self.serve_file(str(file_path))
            else:
                self.send_error(404, "File not found")
        else:
            self.send_error(404, "Page not found")
    
    def do_POST(self):
        """معالجة طلبات POST"""
        if self.path == '/enhance':
            self.handle_enhance_request()
        else:
            self.send_error(404, "Endpoint not found")
    
    def serve_file(self, file_path):
        """تقديم ملف"""
        try:
            if not os.path.isabs(file_path):
                file_path = self.base_path / file_path
            
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # تحديد نوع المحتوى
            content_type, _ = mimetypes.guess_type(str(file_path))
            if content_type is None:
                content_type = 'application/octet-stream'
            
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(content)
            
        except FileNotFoundError:
            self.send_error(404, "File not found")
        except Exception as e:
            print(f"Error serving file: {e}")
            self.send_error(500, "Internal server error")
    
    def handle_enhance_request(self):
        """معالجة طلب تحسين الصورة"""
        try:
            # قراءة البيانات
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # تحليل JSON
            data = json.loads(post_data.decode('utf-8'))
            image_data = data.get('image')
            
            if not image_data:
                self.send_json_response({'error': 'لم يتم العثور على بيانات الصورة'}, 400)
                return
            
            # فك تشفير الصورة من base64
            try:
                # إزالة البادئة data:image/...;base64,
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                
                image_bytes = base64.b64decode(image_data)
            except Exception as e:
                self.send_json_response({'error': f'خطأ في فك تشفير الصورة: {str(e)}'}, 400)
                return
            
            # إنشاء ملف مؤقت للصورة
            temp_dir = self.base_path / 'temp'
            temp_dir.mkdir(exist_ok=True)
            
            input_filename = f"input_{uuid.uuid4().hex}.jpg"
            output_filename = f"output_{uuid.uuid4().hex}.jpg"
            
            input_path = temp_dir / input_filename
            output_path = self.base_path / 'results' / output_filename
            
            # حفظ الصورة المدخلة
            with open(input_path, 'wb') as f:
                f.write(image_bytes)
            
            # تشغيل Real-ESRGAN
            success, error_msg = self.run_realesrgan(str(input_path), str(output_path))
            
            # تنظيف الملف المؤقت
            try:
                input_path.unlink()
            except:
                pass
            
            if success and output_path.exists():
                # إرجاع مسار الصورة المحسنة
                result_url = f'/results/{output_filename}'
                self.send_json_response({
                    'success': True,
                    'enhanced_image_url': result_url,
                    'message': 'تم تحسين الصورة بنجاح!'
                })
            else:
                self.send_json_response({
                    'error': f'فشل في تحسين الصورة: {error_msg}'
                }, 500)
                
        except json.JSONDecodeError:
            self.send_json_response({'error': 'بيانات JSON غير صالحة'}, 400)
        except Exception as e:
            print(f"Error in enhance request: {e}")
            self.send_json_response({'error': f'خطأ في الخادم: {str(e)}'}, 500)
    
    def run_realesrgan(self, input_path, output_path):
        """تشغيل Real-ESRGAN"""
        try:
            # التأكد من وجود المجلد
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # تشغيل Real-ESRGAN NCNN
            exe_path = self.base_path / 'realesrgan-ncnn-vulkan.exe'
            
            if not exe_path.exists():
                return False, "ملف Real-ESRGAN غير موجود"
            
            cmd = [
                str(exe_path),
                '-i', input_path,
                '-o', output_path,
                '-n', 'realesrgan-x4plus',
                '-t', '256',  # تقليل حجم البلاط لتسريع المعالجة
                '-j', '1:1:1'  # استخدام خيط واحد لكل GPU لتحسين الاستقرار
            ]
            
            print(f"Running command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.base_path)
            )
            
            if result.returncode == 0:
                return True, None
            else:
                error_msg = result.stderr or result.stdout or "خطأ غير معروف"
                print(f"Real-ESRGAN error: {error_msg}")
                return False, error_msg
                
        except Exception as e:
            print(f"Exception in run_realesrgan: {e}")
            return False, str(e)
    
    def send_json_response(self, data, status_code=200):
        """إرسال استجابة JSON"""
        response = json.dumps(data, ensure_ascii=False).encode('utf-8')
        
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(response)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(response)
    
    def do_OPTIONS(self):
        """معالجة طلبات OPTIONS للـ CORS"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def log_message(self, format, *args):
        """تسجيل الرسائل"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {format % args}")

def main():
    """تشغيل الخادم"""
    # استخدام متغير البيئة PORT للنشر السحابي أو 8080 للتطوير المحلي
    port = int(os.environ.get('PORT', 8080))
    host = '0.0.0.0'  # للسماح بالاتصالات الخارجية في البيئة السحابية
    
    # التأكد من وجود مجلد النتائج
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # إنشاء الخادم
    server = HTTPServer((host, port), RealESRGANHandler)
    
    print(f"🚀 Real-ESRGAN Web Server بدأ التشغيل على:")
    print(f"   http://{host}:{port}")
    print(f"   اضغط Ctrl+C لإيقاف الخادم")
    print("="*50)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 تم إيقاف الخادم")
        server.shutdown()

if __name__ == '__main__':
    main()