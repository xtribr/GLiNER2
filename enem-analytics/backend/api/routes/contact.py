"""Contact form endpoint"""
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
import resend

router = APIRouter(prefix="/api/contact", tags=["Contact"])

# Configure Resend
resend.api_key = os.getenv("RESEND_API_KEY")

class ContactForm(BaseModel):
    nome: str
    telefone: str
    nome_escola: str
    cargo: str
    conhecia_xtri: bool
    comentarios: str = ""


@router.post("")
async def send_contact(form: ContactForm):
    """Send contact form via email"""
    if not resend.api_key:
        raise HTTPException(status_code=500, detail="Email service not configured")

    try:
        html_content = f"""
        <h2>Novo Contato - X-TRI Escolas</h2>
        <table style="border-collapse: collapse; width: 100%;">
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>Nome</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{form.nome}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>Telefone/WhatsApp</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{form.telefone}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>Escola</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{form.nome_escola}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>Cargo</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{form.cargo}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>Conhecia a XTRI?</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{"Sim" if form.conhecia_xtri else "Não"}</td>
            </tr>
        </table>

        <h3>Comentários</h3>
        <p style="background: #f5f5f5; padding: 12px; border-radius: 8px;">
            {form.comentarios or "Nenhum comentário"}
        </p>
        """

        params = {
            "from": "X-TRI Escolas <contato@xtri.online>",
            "to": ["contato@xtri.online"],
            "subject": f"Novo Contato - {form.nome_escola}",
            "html": html_content,
            "reply_to": form.telefone,
        }

        resend.Emails.send(params)

        return {"success": True, "message": "Mensagem enviada com sucesso"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao enviar email: {str(e)}")
