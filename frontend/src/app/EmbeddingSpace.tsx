"use client";

import React, { useRef, useEffect, useState } from "react";

export interface ChunkNode {
  id: number;
  text: string;
  x: number;
  y: number;
  z: number;
}

interface EmbeddingSpaceProps {
  nodes: ChunkNode[];
  retrievedChunks: string[];
}

interface Star {
  x: number;
  y: number;
  size: number;
  brightness: number;
  speed: number;
}

interface Meteor {
  x: number;
  y: number;
  speedX: number;
  speedY: number;
  length: number;
  size: number;
  opacity: number;
}

export default function EmbeddingSpace({ nodes, retrievedChunks }: EmbeddingSpaceProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [hoveredNode, setHoveredNode] = useState<ChunkNode | null>(null);
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  // Rotation angles for 3D simulation
  const angleXRef = useRef<number>(0.3);
  const angleYRef = useRef<number>(0.5);
  const hoverNodeRef = useRef<ChunkNode | null>(null);
  const isHoveredRef = useRef<boolean>(false);

  // Space assets refs
  const starsRef = useRef<Star[]>([]);
  const meteorsRef = useRef<Meteor[]>([]);

  // Track cursor position for canvas hover detection
  const mousePosRef = useRef<{ x: number; y: number }>({ x: -1000, y: -1000 });

  useEffect(() => {
    // Generate stars
    const stars: Star[] = [];
    for (let i = 0; i < 60; i++) {
      stars.push({
        x: Math.random() * 1920,
        y: Math.random() * 1080,
        size: Math.random() * 1.5 + 0.5,
        brightness: Math.random(),
        speed: Math.random() * 0.1 + 0.05,
      });
    }
    starsRef.current = stars;

    // Generate active meteors
    const meteors: Meteor[] = [];
    for (let i = 0; i < 3; i++) {
      meteors.push({
        x: Math.random() * 1920,
        y: Math.random() * 500,
        speedX: -(Math.random() * 1.2 + 0.8),
        speedY: Math.random() * 0.8 + 0.4,
        length: Math.random() * 40 + 30,
        size: Math.random() * 1.5 + 1.0,
        opacity: Math.random(),
      });
    }
    meteorsRef.current = meteors;
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationFrameId: number;

    // Resize canvas dynamically to container
    const resizeCanvas = () => {
      if (containerRef.current) {
        canvas.width = containerRef.current.clientWidth;
        canvas.height = containerRef.current.clientHeight || 550;
      }
    };
    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    // Render loop
    const render = () => {
      if (!ctx || !canvas) return;
      const width = canvas.width;
      const height = canvas.height;
      const centerX = width / 2;
      const centerY = height / 2;
      const scaleFactor = Math.min(width, height) * 0.4;

      // Draw real space background (pitch black with deep purple glow)
      ctx.fillStyle = "#000000";
      ctx.fillRect(0, 0, width, height);

      // Draw subtle space nebula glow
      const radialGlow = ctx.createRadialGradient(centerX, centerY, 50, centerX, centerY, Math.max(width, height) * 0.8);
      radialGlow.addColorStop(0, "rgba(24, 18, 54, 0.25)");
      radialGlow.addColorStop(1, "rgba(0, 0, 0, 0)");
      ctx.fillStyle = radialGlow;
      ctx.fillRect(0, 0, width, height);

      // Update and Draw Stars
      starsRef.current.forEach((star) => {
        // Move star slowly to left to simulate cosmic motion
        star.x -= star.speed;
        if (star.x < 0) {
          star.x = width;
          star.y = Math.random() * height;
        }

        // Star twinkle
        star.brightness += (Math.random() - 0.5) * 0.05;
        star.brightness = Math.max(0.2, Math.min(1.0, star.brightness));

        ctx.fillStyle = `rgba(255, 255, 255, ${star.brightness})`;
        ctx.beginPath();
        ctx.arc(star.x % width, star.y % height, star.size, 0, Math.PI * 2);
        ctx.fill();
      });

      // Update and Draw Meteors (slow speed)
      meteorsRef.current.forEach((meteor) => {
        meteor.x += meteor.speedX;
        if (meteor.x < -100) {
          meteor.x = width + 100;
          meteor.y = Math.random() * (height * 0.6);
        }
        meteor.y += meteor.speedY;
        if (meteor.y > height + 100) {
          meteor.y = -50;
          meteor.x = Math.random() * width + 200;
        }

        // Draw meteor trail
        const grad = ctx.createLinearGradient(meteor.x, meteor.y, meteor.x - meteor.speedX * meteor.length * 0.3, meteor.y - meteor.speedY * meteor.length * 0.3);
        grad.addColorStop(0, "rgba(255, 255, 255, 0.8)");
        grad.addColorStop(0.2, "rgba(139, 92, 246, 0.4)");
        grad.addColorStop(1, "rgba(0, 0, 0, 0)");

        ctx.strokeStyle = grad;
        ctx.lineWidth = meteor.size;
        ctx.beginPath();
        ctx.moveTo(meteor.x, meteor.y);
        ctx.lineTo(meteor.x + meteor.speedX * meteor.length * 0.5, meteor.y + meteor.speedY * meteor.length * 0.5);
        ctx.stroke();
      });

      // Pause rotation on hover
      if (!isHoveredRef.current) {
        angleYRef.current += 0.0025;
        angleXRef.current = 0.25 + Math.sin(Date.now() * 0.0004) * 0.08;
      }

      const cosX = Math.cos(angleXRef.current);
      const sinX = Math.sin(angleXRef.current);
      const cosY = Math.cos(angleYRef.current);
      const sinY = Math.sin(angleYRef.current);

      // Technical reference grid rings
      ctx.strokeStyle = "rgba(139, 92, 246, 0.04)";
      ctx.lineWidth = 1;
      for (let r = 1; r <= 3; r++) {
        ctx.beginPath();
        ctx.arc(centerX, centerY, (scaleFactor / 3) * r, 0, Math.PI * 2);
        ctx.stroke();
      }

      // Project 3D nodes
      interface ProjectedNode {
        node: ChunkNode;
        projX: number;
        projY: number;
        projZ: number;
        scale: number;
        isRetrieved: boolean;
      }

      const projected: ProjectedNode[] = nodes.map((node) => {
        let x1 = node.x * cosY - node.z * sinY;
        let z1 = node.z * cosY + node.x * sinY;
        let y2 = node.y * cosX - z1 * sinX;
        let z2 = z1 * cosX + node.y * sinX;

        const cameraDistance = 15;
        const perspective = cameraDistance / (cameraDistance + z2);
        const projX = centerX + x1 * scaleFactor * 0.7 * perspective;
        const projY = centerY + y2 * scaleFactor * 0.7 * perspective;

        const isRetrieved = retrievedChunks.some((rc) => 
          rc.toLowerCase().includes(node.text.toLowerCase()) || 
          node.text.toLowerCase().includes(rc.toLowerCase())
        );

        return {
          node,
          projX,
          projY,
          projZ: z2,
          scale: perspective,
          isRetrieved,
        };
      });

      projected.sort((a, b) => b.projZ - a.projZ);

      // Draw connection lines
      ctx.lineWidth = 1.2;
      for (let i = 0; i < projected.length - 1; i++) {
        const from = projected[i];
        const to = projected[i + 1];
        const avgScale = (from.scale + to.scale) / 2;
        const baseOpacity = from.isRetrieved && to.isRetrieved ? 0.45 : 0.08;

        const grad = ctx.createLinearGradient(from.projX, from.projY, to.projX, to.projY);
        grad.addColorStop(0, from.isRetrieved ? `rgba(16, 185, 129, ${0.8 * avgScale})` : `rgba(139, 92, 246, ${baseOpacity * avgScale})`);
        grad.addColorStop(1, to.isRetrieved ? `rgba(16, 185, 129, ${0.8 * avgScale})` : `rgba(139, 92, 246, ${baseOpacity * avgScale})`);

        ctx.strokeStyle = grad;
        ctx.beginPath();
        ctx.moveTo(from.projX, from.projY);
        ctx.lineTo(to.projX, to.projY);
        ctx.stroke();
      }

      // Check mouse hover
      let foundHover: ChunkNode | null = null;
      const mouseX = mousePosRef.current.x;
      const mouseY = mousePosRef.current.y;

      projected.forEach((p) => {
        const nodeSize = p.isRetrieved ? 13 * p.scale : 6.5 * p.scale;
        const dx = mouseX - p.projX;
        const dy = mouseY - p.projY;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < Math.max(nodeSize + 8, 14)) {
          foundHover = p.node;
        }

        // Draw Nodes
        ctx.beginPath();
        ctx.arc(p.projX, p.projY, nodeSize, 0, Math.PI * 2);

        if (p.isRetrieved) {
          ctx.shadowBlur = 18;
          ctx.shadowColor = "#10b981";
          ctx.fillStyle = "#10b981";
        } else {
          ctx.shadowBlur = 0;
          ctx.fillStyle = `rgba(139, 92, 246, ${0.45 + p.scale * 0.55})`;
        }
        ctx.fill();

        ctx.shadowBlur = 0;
        ctx.strokeStyle = p.isRetrieved ? "rgba(255, 255, 255, 0.95)" : `rgba(139, 92, 246, ${0.3 * p.scale})`;
        ctx.lineWidth = p.isRetrieved ? 2.5 : 1;
        ctx.beginPath();
        ctx.arc(p.projX, p.projY, nodeSize + (p.isRetrieved ? 4 : 2), 0, Math.PI * 2);
        ctx.stroke();

        if (p.isRetrieved) {
          ctx.fillStyle = "#ffffff";
          ctx.font = "bold 9.5px var(--font-sans)";
          ctx.textAlign = "center";
          ctx.fillText(`Chunk ${p.node.id}`, p.projX, p.projY - nodeSize - 8);
        }
      });

      if (foundHover !== hoverNodeRef.current) {
        hoverNodeRef.current = foundHover;
        setHoveredNode(foundHover);
      }

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      window.removeEventListener("resize", resizeCanvas);
      cancelAnimationFrame(animationFrameId);
    };
  }, [nodes, retrievedChunks]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    mousePosRef.current = { x, y };
    isHoveredRef.current = true;
    setTooltipPos({ x: e.clientX - rect.left + 20, y: e.clientY - rect.top + 10 });
  };

  const handleMouseLeave = () => {
    mousePosRef.current = { x: -1000, y: -1000 };
    isHoveredRef.current = false;
    setHoveredNode(null);
    hoverNodeRef.current = null;
  };

  return (
    <div ref={containerRef} style={{ width: "100%", height: "100%", position: "relative" }}>
      <canvas
        ref={canvasRef}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        style={{ display: "block", borderRadius: "14px", cursor: hoveredNode ? "pointer" : "default" }}
      />
      
      {hoveredNode && (
        <div
          style={{
            position: "absolute",
            left: `${tooltipPos.x}px`,
            top: `${tooltipPos.y}px`,
            background: "rgba(18, 14, 32, 0.75)",
            backdropFilter: "blur(20px)",
            WebkitBackdropFilter: "blur(20px)",
            color: "#f3f4f6",
            padding: "16px",
            borderRadius: "14px",
            fontSize: "12.5px",
            lineHeight: "1.5",
            maxWidth: "340px",
            zIndex: 100,
            border: "1px solid rgba(255, 255, 255, 0.08)",
            boxShadow: "0 12px 40px 0 rgba(0, 0, 0, 0.55), inset 0 0 15px rgba(255, 255, 255, 0.05)",
            pointerEvents: "none",
            fontFamily: "var(--font-sans)",
            transition: "all 0.15s ease-out",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
            <strong style={{ color: "#a78bfa", fontFamily: "var(--font-title)", fontSize: "14px", fontWeight: "700" }}>
              Semantic Chunk {hoveredNode.id}
            </strong>
            <span style={{ fontSize: "10px", color: "var(--accent-secondary)", background: "rgba(236, 72, 153, 0.15)", padding: "2px 6px", borderRadius: "4px", fontWeight: "600" }}>
              x: {hoveredNode.x.toFixed(2)} y: {hoveredNode.y.toFixed(2)}
            </span>
          </div>
          <span style={{ color: "#d1d5db", display: "block", fontStyle: "normal", fontSize: "12px", maxHeight: "150px", overflowY: "hidden" }}>
            "{hoveredNode.text.slice(0, 200)}
            {hoveredNode.text.length > 200 ? "..." : ""}"
          </span>
        </div>
      )}
    </div>
  );
}
