#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sistema de Alertas para Detección de Anomalías
==============================================

Este módulo proporciona un sistema de alertas para notificar cuando se detecta
una anomalía en el proceso de monitoreo. Admite múltiples métodos de notificación:
- Registro local de eventos
- Notificaciones por correo electrónico
- Mensajes de texto (SMS)
- Notificaciones push a través de aplicaciones de mensajería
- Integración con sistemas de gestión de Veolia Kruger

El sistema incluye funciones para evitar alertas repetitivas y mantener
un registro de incidentes para análisis posterior.
"""

import os
import sys
import json
import time
import logging
import smtplib
import threading
import requests
import cv2
import numpy as np
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime, timedelta
from pathlib import Path

# Configurar logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("anomaly_alerts.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("AnomalyAlertSystem")


class AlertSystem:
    """Sistema de alertas para detección de anomalías."""
    
    def __init__(self, config_file=None, enabled=True, cooldown_period=300):
        """
        Inicializa el sistema de alertas.
        
        Args:
            config_file: Ruta al archivo de configuración JSON (opcional)
            enabled: Estado inicial del sistema (habilitado/deshabilitado)
            cooldown_period: Período de enfriamiento entre alertas repetidas (segundos)
        """
        self.enabled = enabled
        self.cooldown_period = cooldown_period
        self.last_alert_time = 0
        self.alert_count = 0
        self.alert_history = []
        
        # Cargar configuración
        self.config = {
            "email": {
                "enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "",
                "password": "",
                "from_addr": "",
                "to_addr": []
            },
            "sms": {
                "enabled": False,
                "api_key": "",
                "phone_numbers": []
            },
            "push": {
                "enabled": False,
                "api_endpoint": "",
                "api_key": "",
                "recipients": []
            },
            "veolia_integration": {
                "enabled": False,
                "api_endpoint": "",
                "api_key": "",
                "system_id": ""
            },
            "log_dir": "alert_logs"
        }
        
        # Sobrescribir con configuración del archivo si existe
        if config_file:
            self.load_config(config_file)
            
        # Crear directorio de logs si no existe
        self.log_dir = self.config["log_dir"]
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Inicializar métodos de alerta
        self._init_alert_methods()
        
        logger.info("Sistema de alertas inicializado")
    
    def load_config(self, config_file):
        """
        Carga la configuración desde un archivo JSON.
        
        Args:
            config_file: Ruta al archivo de configuración
        """
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                
            # Actualizar configuración manteniendo valores predeterminados para claves faltantes
            for key, value in config.items():
                if key in self.config:
                    if isinstance(self.config[key], dict) and isinstance(value, dict):
                        self.config[key].update(value)
                    else:
                        self.config[key] = value
                        
            logger.info(f"Configuración cargada desde {config_file}")
        except Exception as e:
            logger.error(f"Error al cargar configuración: {e}")
    
    def _init_alert_methods(self):
        """Inicializa los diferentes métodos de alerta según la configuración."""
        # Aquí se inicializarían conexiones, autenticación, etc.
        pass
    
    def enable(self):
        """Habilita el sistema de alertas."""
        self.enabled = True
        logger.info("Sistema de alertas habilitado")
    
    def disable(self):
        """Deshabilita el sistema de alertas."""
        self.enabled = False
        logger.info("Sistema de alertas deshabilitado")
    
    def should_send_alert(self, severity="high"):
        """
        Determina si se debe enviar una alerta basado en el período de enfriamiento.
        
        Args:
            severity: Nivel de severidad de la alerta
            
        Returns:
            True si se debe enviar la alerta, False en caso contrario
        """
        current_time = time.time()
        
        # Siempre enviar para alertas críticas
        if severity == "critical":
            return True
            
        # Verificar período de enfriamiento para otras alertas
        time_since_last = current_time - self.last_alert_time
        
        if time_since_last >= self.cooldown_period:
            return True
        else:
            logger.debug(f"Alerta en enfriamiento. Tiempo restante: {self.cooldown_period - time_since_last:.1f}s")
            return False
    
    def send_alert(self, image, anomaly_score, severity="high", details=None):
        """
        Envía una alerta por todos los canales configurados.
        
        Args:
            image: Imagen donde se detectó la anomalía
            anomaly_score: Puntuación de anomalía (0-1)
            severity: Nivel de severidad ("low", "medium", "high", "critical")
            details: Detalles adicionales para incluir en la alerta
        
        Returns:
            True si la alerta fue enviada, False en caso contrario
        """
        if not self.enabled:
            logger.debug("Sistema de alertas deshabilitado. Alerta no enviada.")
            return False
            
        if not self.should_send_alert(severity):
            return False
            
        # Generar ID de alerta y timestamp
        alert_id = f"ANM-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        timestamp = datetime.now().isoformat()
        
        # Crear objeto de alerta
        alert = {
            "id": alert_id,
            "timestamp": timestamp,
            "anomaly_score": float(anomaly_score),
            "severity": severity,
            "details": details or {}
        }
        
        # Registrar alerta
        self._log_alert(alert, image)
        
        # Enviar por correo si está habilitado
        if self.config["email"]["enabled"]:
            self._send_email_alert(alert, image)
            
        # Enviar SMS si está habilitado
        if self.config["sms"]["enabled"]:
            self._send_sms_alert(alert)
            
        # Enviar notificación push si está habilitado
        if self.config["push"]["enabled"]:
            self._send_push_alert(alert)
            
        # Integración con sistemas de Veolia
        if self.config["veolia_integration"]["enabled"]:
            self._send_veolia_alert(alert, image)
        
        # Actualizar estado
        self.last_alert_time = time.time()
        self.alert_count += 1
        self.alert_history.append(alert)
        
        logger.info(f"Alerta {alert_id} enviada. Anomalía: {anomaly_score:.4f}, Severidad: {severity}")
        return True
    
    def _log_alert(self, alert, image):
        """
        Registra una alerta en el sistema local.
        
        Args:
            alert: Información de la alerta
            image: Imagen donde se detectó la anomalía
        """
        try:
            # Guardar imagen
            img_filename = os.path.join(self.log_dir, f"{alert['id']}.jpg")
            cv2.imwrite(img_filename, image)
            
            # Guardar información de la alerta como JSON
            json_filename = os.path.join(self.log_dir, f"{alert['id']}.json")
            with open(json_filename, 'w') as f:
                json.dump(alert, f, indent=4)
                
            logger.debug(f"Alerta registrada en {json_filename}")
        except Exception as e:
            logger.error(f"Error al registrar alerta: {e}")
    
    def _send_email_alert(self, alert, image):
        """
        Envía una alerta por correo electrónico.
        
        Args:
            alert: Información de la alerta
            image: Imagen donde se detectó la anomalía
        """
        if not self.config["email"]["enabled"] or not self.config["email"]["to_addr"]:
            return
            
        try:
            # Crear mensaje
            msg = MIMEMultipart()
            msg['Subject'] = f"ALERTA: Anomalía detectada [{alert['id']}]"
            msg['From'] = self.config["email"]["from_addr"]
            msg['To'] = ", ".join(self.config["email"]["to_addr"])
            
            # Cuerpo del mensaje
            body = f"""
            <html>
            <body>
                <h2>Alerta de Anomalía Detectada</h2>
                <p><strong>ID de Alerta:</strong> {alert['id']}</p>
                <p><strong>Timestamp:</strong> {alert['timestamp']}</p>
                <p><strong>Puntuación de Anomalía:</strong> {alert['anomaly_score']:.4f}</p>
                <p><strong>Severidad:</strong> {alert['severity'].upper()}</p>
                <hr>
                <p>Sistema automático de detección de anomalías - Veolia Kruger</p>
                <p>Por favor, no responda a este correo.</p>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            # Adjuntar imagen
            img_data = cv2.imencode('.jpg', image)[1].tostring()
            img_attachment = MIMEImage(img_data, name=f"{alert['id']}.jpg")
            msg.attach(img_attachment)
            
            # Enviar correo
            server = smtplib.SMTP(self.config["email"]["smtp_server"], self.config["email"]["smtp_port"])
            server.starttls()
            server.login(self.config["email"]["username"], self.config["email"]["password"])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Alerta enviada por correo a {len(self.config['email']['to_addr'])} destinatarios")
        except Exception as e:
            logger.error(f"Error al enviar alerta por correo: {e}")
    
    def _send_sms_alert(self, alert):
        """
        Envía una alerta por SMS.
        
        Args:
            alert: Información de la alerta
        """
        if not self.config["sms"]["enabled"] or not self.config["sms"]["phone_numbers"]:
            return
            
        try:
            # Contenido del mensaje
            message = (f"ALERTA: Anomalía detectada. "
                      f"ID: {alert['id']}. "
                      f"Puntuación: {alert['anomaly_score']:.4f}. "
                      f"Severidad: {alert['severity'].upper()}")
            
            # Este es un ejemplo de integración con un servicio de SMS
            # En una implementación real, aquí se utilizaría la API del proveedor de SMS
            for phone in self.config["sms"]["phone_numbers"]:
                logger.info(f"Simulando envío de SMS a {phone}: {message}")
                
                # Ejemplo de código para un servicio real (comentado)
                """
                response = requests.post(
                    "https://api.sms-service.com/send",
                    headers={"Authorization": f"Bearer {self.config['sms']['api_key']}"},
                    json={
                        "to": phone,
                        "message": message
                    }
                )
                response.raise_for_status()
                """
            
            logger.info(f"Alerta enviada por SMS a {len(self.config['sms']['phone_numbers'])} números")
        except Exception as e:
            logger.error(f"Error al enviar alerta por SMS: {e}")
    
    def _send_push_alert(self, alert):
        """
        Envía una notificación push.
        
        Args:
            alert: Información de la alerta
        """
        if not self.config["push"]["enabled"]:
            return
            
        try:
            # Datos de la notificación
            notification_data = {
                "title": "Anomalía Detectada",
                "body": f"ID: {alert['id']}. Puntuación: {alert['anomaly_score']:.4f}. Severidad: {alert['severity'].upper()}",
                "data": {
                    "alert_id": alert['id'],
                    "timestamp": alert['timestamp'],
                    "score": alert['anomaly_score'],
                    "severity": alert['severity']
                },
                "recipients": self.config["push"]["recipients"]
            }
            
            # Ejemplo de integración con un servicio de notificaciones push
            logger.info(f"Simulando envío de notificación push: {notification_data}")
            
            # Código para un servicio real (comentado)
            """
            response = requests.post(
                self.config["push"]["api_endpoint"],
                headers={
                    "Authorization": f"Bearer {self.config['push']['api_key']}",
                    "Content-Type": "application/json"
                },
                json=notification_data
            )
            response.raise_for_status()
            """
            
            logger.info(f"Alerta enviada como notificación push")
        except Exception as e:
            logger.error(f"Error al enviar notificación push: {e}")
    
    def _send_veolia_alert(self, alert, image):
        """
        Integra con sistemas internos de Veolia Kruger.
        
        Args:
            alert: Información de la alerta
            image: Imagen donde se detectó la anomalía
        """
        if not self.config["veolia_integration"]["enabled"]:
            return
            
        try:
            # Preparar datos para el sistema de Veolia
            integration_data = {
                "system_id": self.config["veolia_integration"]["system_id"],
                "alert_type": "anomaly_detection",
                "alert_id": alert['id'],
                "timestamp": alert['timestamp'],
                "severity": alert['severity'],
                "metrics": {
                    "anomaly_score": alert['anomaly_score']
                },
                "details": alert['details']
            }
            
            # Simular integración con sistema de Veolia
            logger.info(f"Simulando integración con sistema Veolia: {integration_data}")
            
            # Código para integración real (comentado)
            """
            # Preparar imagen para envío
            files = {
                'image': (f"{alert['id']}.jpg", cv2.imencode('.jpg', image)[1].tostring(), 'image/jpeg')
            }
            
            # Enviar datos al endpoint de Veolia
            response = requests.post(
                self.config["veolia_integration"]["api_endpoint"],
                headers={"Authorization": f"Bearer {self.config['veolia_integration']['api_key']}"},
                data={"data": json.dumps(integration_data)},
                files=files
            )
            response.raise_for_status()
            """
            
            logger.info(f"Alerta enviada al sistema de Veolia Kruger")
        except Exception as e:
            logger.error(f"Error en integración con sistema Veolia: {e}")
    
    def get_alert_history(self, limit=10):
        """
        Obtiene el historial de alertas recientes.
        
        Args:
            limit: Número máximo de alertas a devolver
            
        Returns:
            Lista de alertas recientes
        """
        return self.alert_history[-limit:]
    
    def get_alert_stats(self):
        """
        Obtiene estadísticas de alertas.
        
        Returns:
            Diccionario con estadísticas
        """
        stats = {
            "total_alerts": self.alert_count,
            "alerts_today": 0,
            "alerts_this_week": 0,
            "alerts_by_severity": {
                "low": 0,
                "medium": 0,
                "high": 0,
                "critical": 0
            }
        }
        
        # Calcular estadísticas basadas en el historial
        now = datetime.now()
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = today - timedelta(days=today.weekday())
        
        for alert in self.alert_history:
            # Parsear timestamp
            try:
                alert_time = datetime.fromisoformat(alert["timestamp"])
                
                # Contar por período
                if alert_time >= today:
                    stats["alerts_today"] += 1
                    
                if alert_time >= week_start:
                    stats["alerts_this_week"] += 1
                
                # Contar por severidad
                severity = alert["severity"]
                if severity in stats["alerts_by_severity"]:
                    stats["alerts_by_severity"][severity] += 1
            except (ValueError, KeyError):
                continue
                
        return stats
    
    def close(self):
        """Cierra conexiones y recursos del sistema de alertas."""
        logger.info("Cerrando sistema de alertas")
        # Cerrar conexiones, liberar recursos, etc.


# Función para crear una configuración de ejemplo
def create_example_config(output_path="alert_config.json"):
    """
    Crea un archivo de configuración de ejemplo para el sistema de alertas.
    
    Args:
        output_path: Ruta donde guardar el archivo de configuración
    """
    config = {
        "email": {
            "enabled": False,
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "username": "tu_correo@gmail.com",
            "password": "tu_contraseña",
            "from_addr": "alertas@veolia-kruger.com",
            "to_addr": ["operador1@veolia-kruger.com", "supervisor@veolia-kruger.com"]
        },
        "sms": {
            "enabled": False,
            "api_key": "tu_api_key",
            "phone_numbers": ["+1234567890", "+0987654321"]
        },
        "push": {
            "enabled": False,
            "api_endpoint": "https://push-service.example.com/send",
            "api_key": "tu_api_key",
            "recipients": ["user1", "user2"]
        },
        "veolia_integration": {
            "enabled": False,
            "api_endpoint": "https://api.veolia-kruger.com/alerts",
            "api_key": "tu_api_key",
            "system_id": "sludge-monitoring-001"
        },
        "log_dir": "alert_logs"
    }
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)
        
    print(f"Configuración de ejemplo guardada en {output_path}")


# Ejemplo de uso
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--create-config":
        create_example_config()
        sys.exit(0)
        
    # Inicializar sistema de alertas
    alert_system = AlertSystem()
    
    # Simular detección de anomalía
    print("Simulando detección de anomalía...")
    
    # Crear imagen de ejemplo
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "ANOMALIA SIMULADA", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Enviar alerta
    alert_system.send_alert(img, 0.85, severity="high", 
                          details={"location": "Línea de salida principal"})
    
    # Mostrar estadísticas
    print("\nEstadísticas de alertas:")
    stats = alert_system.get_alert_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    # Cerrar sistema
    alert_system.close()