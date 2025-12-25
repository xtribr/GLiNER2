import { test, expect } from '@playwright/test';

test.describe('Authentication', () => {
  test.beforeEach(async ({ page }) => {
    // Clear localStorage before each test
    await page.goto('/login');
    await page.evaluate(() => localStorage.clear());
  });

  test('should show login page', async ({ page }) => {
    await page.goto('/login');
    await expect(page.getByRole('heading', { name: 'Entrar na sua conta' })).toBeVisible();
    await expect(page.getByPlaceholder('seu@email.com')).toBeVisible();
    await expect(page.getByPlaceholder('Sua senha')).toBeVisible();
  });

  test('should show error for invalid credentials', async ({ page }) => {
    await page.goto('/login');
    await page.fill('input[type="email"]', 'wrong@email.com');
    await page.fill('input[type="password"]', 'wrongpass');
    await page.click('button[type="submit"]');

    await expect(page.getByText('Email ou senha incorretos')).toBeVisible({ timeout: 10000 });
  });

  test('should redirect to login when not authenticated', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveURL('/login', { timeout: 10000 });
  });
});

test.describe('Super Admin', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/login');
    await page.evaluate(() => localStorage.clear());

    // Login as super admin
    await page.fill('input[type="email"]', 'super@xtri.com');
    await page.fill('input[type="password"]', 'xtri2025');
    await page.click('button[type="submit"]');

    // Wait for redirect
    await page.waitForURL('/', { timeout: 10000 });
  });

  test('should access dashboard', async ({ page }) => {
    await expect(page).toHaveURL('/');
  });

  test('should see admin menu', async ({ page }) => {
    await expect(page.getByText('Painel Admin')).toBeVisible();
    await expect(page.getByText('Usuários')).toBeVisible();
  });

  test('should access schools list', async ({ page }) => {
    await page.goto('/schools');
    await expect(page).toHaveURL('/schools');
  });

  test('should access admin panel', async ({ page }) => {
    await page.click('text=Painel Admin');
    await expect(page).toHaveURL('/admin');
    await expect(page.getByText('Painel Administrativo')).toBeVisible();
  });

  test('should access users management', async ({ page }) => {
    await page.click('text=Usuários');
    await expect(page).toHaveURL('/admin/users');
    await expect(page.getByText('Gerenciar Escolas')).toBeVisible();
  });

  test('should logout', async ({ page }) => {
    await page.click('text=Sair');
    await expect(page).toHaveURL('/login', { timeout: 10000 });
  });
});

test.describe('School User', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/login');
    await page.evaluate(() => localStorage.clear());

    // Login as school user
    await page.fill('input[type="email"]', 'escola@teste.com');
    await page.fill('input[type="password"]', 'teste123');
    await page.click('button[type="submit"]');

    // Wait for redirect to school page
    await page.waitForURL('/schools/31000000', { timeout: 10000 });
  });

  test('should redirect to own school page', async ({ page }) => {
    await expect(page).toHaveURL('/schools/31000000');
  });

  test('should NOT see admin menu', async ({ page }) => {
    await expect(page.getByText('Painel Admin')).not.toBeVisible();
    await expect(page.getByText('ADMIN')).not.toBeVisible();
  });

  test('should see school menu', async ({ page }) => {
    await expect(page.getByText('MINHA ESCOLA')).toBeVisible();
    await expect(page.getByText('Painel')).toBeVisible();
    await expect(page.getByText('Roadmap')).toBeVisible();
  });

  test('should be redirected when trying to access other routes', async ({ page }) => {
    await page.goto('/schools');
    await expect(page).toHaveURL('/schools/31000000', { timeout: 10000 });
  });

  test('should be redirected when trying to access admin', async ({ page }) => {
    await page.goto('/admin');
    await expect(page).toHaveURL('/schools/31000000', { timeout: 10000 });
  });

  test('should be redirected when trying to access another school', async ({ page }) => {
    await page.goto('/schools/99999999');
    await expect(page).toHaveURL('/schools/31000000', { timeout: 10000 });
  });
});
