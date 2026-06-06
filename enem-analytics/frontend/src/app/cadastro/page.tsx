import Link from 'next/link';

// Stub da rota pública /cadastro. O funil completo (escola → conta → pronto)
// entra na Fase 3; por ora garante que os CTAs da vitrine não dão 404.
export default function CadastroPage() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-[#f5fbff] p-6">
      <div className="w-full max-w-md rounded-3xl border border-[#28B7ED]/20 bg-white p-8 text-center shadow-xl shadow-[#28B7ED]/10">
        <div className="mx-auto mb-5 flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-[#28B7ED] to-[#FF4B2E] text-2xl font-black text-white">
          ✕
        </div>
        <h1 className="text-2xl font-black tracking-tight text-slate-950">Cadastre sua escola</h1>
        <p className="mt-3 text-sm leading-6 text-slate-500">
          O cadastro self-service (escola → conta → acesso na hora) entra na próxima fase.
          Por enquanto, explore o ranking público.
        </p>
        <div className="mt-6 flex flex-col gap-3">
          <Link
            href="/"
            className="inline-flex items-center justify-center rounded-2xl bg-gradient-to-r from-[#28B7ED] to-[#FF4B2E] px-5 py-3 text-sm font-bold text-white shadow-lg shadow-[#28B7ED]/20 transition hover:brightness-105"
          >
            Voltar à vitrine
          </Link>
          <Link href="/login" className="text-sm font-semibold text-[#139ED3] hover:underline">
            Já tenho conta · Entrar
          </Link>
        </div>
      </div>
    </div>
  );
}
