<script setup>
import { useRoute } from 'vue-router'
import logoUrl from './assets/logo.png'

const route = useRoute()
</script>

<template>
  <el-container class="app-shell">
    <el-header class="topbar">
      <div class="brand">
        <img class="brand-logo" :src="logoUrl" alt="餐厅 Logo" width="42" height="42" />
        <div>
          <strong>智慧食堂菜单系统</strong>
          <span>菜单推荐与数据管理</span>
        </div>
      </div>
      <el-menu
        :default-active="route.path"
        mode="horizontal"
        router
        :ellipsis="false"
        class="nav-menu"
      >
        <el-menu-item index="/">
          <el-icon><Dish /></el-icon>
          菜单推荐
        </el-menu-item>
        <el-menu-item index="/data-update">
          <el-icon><UploadFilled /></el-icon>
          数据更新
        </el-menu-item>
      </el-menu>
    </el-header>
    <el-main class="main-content">
      <router-view />
    </el-main>
  </el-container>
</template>
