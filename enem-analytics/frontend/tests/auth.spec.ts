import { test, expect } from '@playwright/test';

const adminEmail = process.env.E2E_ADMIN_EMAIL;
const adminPassword = process.env.E2E_ADMIN_PASSWORD;
const schoolEmail = process.env.E2E_SCHOOL_EMAIL;
const schoolPassword = process.env.E2E_SCHOOL_PASSWORD;
const schoolInep = process.env.E2E_SCHOOL_INEP;
const otherSchoolInep = process.env.E2E_OTHER_SCHOOL_INEP;

test.describe('Rotas públicas e autenticação', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/login');
    await page.evaluate(() => localStorage.clear());
  });

  test('exibe a página de login', async ({ page }) => {
    await expect(page.getByRole('heading', { name: 'Entrar na sua conta' })).toBeVisible();
    await expect(page.getByPlaceholder('seu@email.com')).toBeVisible();
    await expect(page.getByPlaceholder('Sua senha')).toBeVisible();
  });

  for (const path of ['/', '/schools', '/redacao']) {
    test(`${path} permanece pública sem sessão`, async ({ page }) => {
      await page.goto(path);
      await expect(page).toHaveURL(new RegExp(`${path === '/' ? '/$' : `${path}$`}`));
    });
  }

  test('redireciona rota profunda para login sem sessão', async ({ page }) => {
    await page.goto('/compare');
    await expect(page).toHaveURL('/login', { timeout: 10000 });
  });
});

test.describe('Validação fail-closed do perfil', () => {
  test.skip(!adminEmail || !adminPassword, 'Defina E2E_ADMIN_EMAIL e E2E_ADMIN_PASSWORD.');

  test('bloqueia o painel durante indisponibilidade e recupera por retry', async ({ page }) => {
    await page.route('**/api/auth/me', async (route) => {
      await route.fulfill({ status: 503, body: '{"detail":"indisponível"}' });
    });

    await page.goto('/login');
    await page.fill('input[type="email"]', adminEmail!);
    await page.fill('input[type="password"]', adminPassword!);
    await page.click('button[type="submit"]');

    await expect(page.getByRole('heading', { name: 'Não foi possível validar seu acesso' })).toBeVisible();

    await page.unroute('**/api/auth/me');
    await page.getByRole('button', { name: 'Tentar novamente' }).click();
    await expect(page).toHaveURL('/', { timeout: 10000 });
  });
});

test.describe('Administrador', () => {
  test.skip(!adminEmail || !adminPassword, 'Defina E2E_ADMIN_EMAIL e E2E_ADMIN_PASSWORD.');

  test.beforeEach(async ({ page }) => {
    await page.goto('/login');
    await page.fill('input[type="email"]', adminEmail!);
    await page.fill('input[type="password"]', adminPassword!);
    await page.click('button[type="submit"]');
    await page.waitForURL('/', { timeout: 10000 });
  });

  test('acessa o painel administrativo', async ({ page }) => {
    await page.goto('/admin');
    await expect(page).toHaveURL('/admin');
    await expect(page.getByText('Painel Administrativo')).toBeVisible();
  });

  test('acessa a gestão de usuários', async ({ page }) => {
    await page.goto('/admin/users');
    await expect(page).toHaveURL('/admin/users');
    await expect(page.getByText('Gerenciar Escolas')).toBeVisible();
  });
});

test.describe('Usuário de escola', () => {
  test.skip(
    !schoolEmail || !schoolPassword || !schoolInep || !otherSchoolInep,
    'Defina as credenciais E2E da escola e dois códigos INEP reais.',
  );

  test.beforeEach(async ({ page }) => {
    await page.goto('/login');
    await page.fill('input[type="email"]', schoolEmail!);
    await page.fill('input[type="password"]', schoolPassword!);
    await page.click('button[type="submit"]');
    await page.waitForURL(`/schools/${schoolInep}`, { timeout: 10000 });
  });

  test('acessa o próprio painel e o ranking público', async ({ page }) => {
    await expect(page).toHaveURL(`/schools/${schoolInep}`);
    await page.goto('/schools');
    await expect(page).toHaveURL('/schools');
  });

  test('não acessa a área administrativa', async ({ page }) => {
    await page.goto('/admin');
    await expect(page).toHaveURL(`/schools/${schoolInep}`, { timeout: 10000 });
  });

  test('não acessa o painel profundo de outra escola', async ({ page }) => {
    await page.goto(`/schools/${otherSchoolInep}`);
    await expect(page).toHaveURL(`/schools/${schoolInep}`, { timeout: 10000 });
  });
});
