import { createRouter, createWebHistory } from 'vue-router'
import RecommendationView from '@/views/RecommendationView.vue'
import DataUpdateView from '@/views/DataUpdateView.vue'

export const routes = [
  {
    path: '/',
    name: 'recommendation',
    component: RecommendationView,
    meta: { title: '菜单推荐' },
  },
  {
    path: '/data-update',
    name: 'data-update',
    component: DataUpdateView,
    meta: { title: '数据更新' },
  },
]

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes,
})

router.afterEach((to) => {
  document.title = `${to.meta.title || '智慧食堂'}｜智慧食堂菜单系统`
})

export default router
