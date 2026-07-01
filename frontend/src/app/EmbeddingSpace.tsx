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

export default function EmbeddingSpace({ nodes, retrievedChunks }: EmbeddingSpaceProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [hoveredNode, setHoveredNode] = useState<ChunkNode | null>(null);
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  // Rotation angles for 3D simulation
  const angleXRef = useRef<number>(0.3);
  const angleYRef = useRef<number>(0.5);
  const hoverNodeRef = useRef<ChunkNode | null>(null);

  // Track cursor position for canvas hover detection
  const mousePosRef = useRef<{ x: number; y: number }>({ x: -1000, y: -1000 });

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
        canvas.height = containerRef.current.clientHeight || 450;
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

      // Slow idle rotation in 3D space
      angleYRef.current += 0.003;
      angleXRef.current = 0.2 + Math.sin(Date.now() * 0.0005) * 0.1; // gentle bobbing

      const cosX = Math.cos(angleXRef.current);
      const sinX = Math.sin(angleXRef.current);
      const cosY = Math.cos(angleYRef.current);
      const sinY = Math.sin(angleYRef.current);

      // Clear with dark tech space theme gradient
      const bgGrad = ctx.createRadialGradient(centerX, centerY, 10, centerX, centerY, Math.max(width, height));
      bgGrad.addColorStop(0, "#111827");
      bgGrad.addColorStop(1, "#030712");
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, 0, width, height);

      // Draw Grid/Space Structure lines to make it look premium and technical
      ctx.strokeStyle = "rgba(59, 130, 246, 0.05)";
      ctx.lineWidth = 1;
      // Drawing longitudinal rings or bounding lines
      for (let r = 1; r <= 3; r++) {
        ctx.beginPath();
        ctx.arc(centerX, centerY, (scaleFactor / 3) * r, 0, Math.PI * 2);
        ctx.stroke();
      }

      // Project 3D nodes to 2D screen coordinates
      interface ProjectedNode {
        node: ChunkNode;
        projX: number;
        projY: number;
        projZ: number;
        scale: number;
        isRetrieved: boolean;
      }

      const projected: ProjectedNode[] = nodes.map((node) => {
        // Translate coordinates to center and apply 3D rotation
        // Y-axis rotation
        let x1 = node.x * cosY - node.z * sinY;
        let z1 = node.z * cosY + node.x * sinY;

        // X-axis rotation
        let y2 = node.y * cosX - z1 * sinX;
        let z2 = z1 * cosX + node.y * sinX;

        // Camera perspective model
        const cameraDistance = 15;
        const perspective = cameraDistance / (cameraDistance + z2);

        const projX = centerX + x1 * scaleFactor * 0.7 * perspective;
        const projY = centerY + y2 * scaleFactor * 0.7 * perspective;

        // Check if this chunk is one of the retrieved ones
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

      // Sort by depth (Z desc) so closer elements render on top
      projected.sort((a, b) => b.projZ - a.projZ);

      // Draw connections between sequential chunks to show document flow
      ctx.lineWidth = 1.5;
      for (let i = 0; i < projected.length - 1; i++) {
        const from = projected[i];
        const to = projected[i + 1];
        
        // Calculate average scale for opacity
        const avgScale = (from.scale + to.scale) / 2;
        const baseOpacity = from.isRetrieved && to.isRetrieved ? 0.4 : 0.08;
        
        const grad = ctx.createLinearGradient(from.projX, from.projY, to.projX, to.projY);
        grad.addColorStop(0, from.isRetrieved ? `rgba(16, 185, 129, ${0.8 * avgScale})` : `rgba(59, 130, 246, ${baseOpacity * avgScale})`);
        grad.addColorStop(1, to.isRetrieved ? `rgba(16, 185, 129, ${0.8 * avgScale})` : `rgba(59, 130, 246, ${baseOpacity * avgScale})`);
        
        ctx.strokeStyle = grad;
        ctx.beginPath();
        ctx.moveTo(from.projX, from.projY);
        ctx.lineTo(to.projX, to.projY);
        ctx.stroke();
      }

      // Check mouse collision with nodes for tooltip interaction
      let foundHover: ChunkNode | null = null;
      const mouseX = mousePosRef.current.x;
      const mouseY = mousePosRef.current.y;

      projected.forEach((p) => {
        const nodeSize = p.isRetrieved ? 12 * p.scale : 6 * p.scale;
        const dx = mouseX - p.projX;
        const dy = mouseY - p.projY;
        const dist = Math.sqrt(dx * dx + dy * dy);

        // Hover detection within 15px radius
        if (dist < Math.max(nodeSize + 6, 12)) {
          foundHover = p.node;
        }

        // --- Draw Node ---
        ctx.beginPath();
        ctx.arc(p.projX, p.projY, nodeSize, 0, Math.PI * 2);

        if (p.isRetrieved) {
          // Glow effect for retrieved nodes
          ctx.shadowBlur = 15;
          ctx.shadowColor = "#10b981";
          ctx.fillStyle = "#10b981";
        } else {
          ctx.shadowBlur = 0;
          ctx.fillStyle = `rgba(59, 130, 246, ${0.4 + p.scale * 0.6})`;
        }
        ctx.fill();

        // Draw outer ring
        ctx.shadowBlur = 0;
        ctx.strokeStyle = p.isRetrieved 
          ? `rgba(255, 255, 255, ${0.8 * p.scale})` 
          : `rgba(99, 102, 241, ${0.3 * p.scale})`;
        ctx.lineWidth = p.isRetrieved ? 2 : 1;
        ctx.beginPath();
        ctx.arc(p.projX, p.projY, nodeSize + (p.isRetrieved ? 4 : 2), 0, Math.PI * 2);
        ctx.stroke();

        // Draw labels for retrieved nodes
        if (p.isRetrieved) {
          ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
          ctx.font = "bold 9px monospace";
          ctx.textAlign = "center";
          ctx.fillText(`Chunk ${p.node.id}`, p.projX, p.projY - nodeSize - 8);
        }
      });

      // Update hovered state
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
    setTooltipPos({ x: e.clientX - rect.left + 15, y: e.clientY - rect.top + 15 });
  };

  const handleMouseLeave = () => {
    mousePosRef.current = { x: -1000, y: -1000 };
    setHoveredNode(null);
    hoverNodeRef.current = null;
  };

  return (
    <div ref={containerRef} style={{ width: "100%", height: "100%", position: "relative" }}>
      <canvas
        ref={canvasRef}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        style={{ display: "block", borderRadius: "8px", cursor: hoveredNode ? "pointer" : "default" }}
      />
      
      {hoveredNode && (
        <div
          style={{
            position: "absolute",
            left: `${tooltipPos.x}px`,
            top: `${tooltipPos.y}px`,
            backgroundColor: "rgba(17, 24, 39, 0.95)",
            color: "#e5e7eb",
            padding: "10px 14px",
            borderRadius: "6px",
            fontSize: "12px",
            lineHeight: "1.4",
            maxWidth: "280px",
            zIndex: 100,
            border: "1px solid rgba(59, 130, 246, 0.3)",
            boxShadow: "0 10px 15px -3px rgba(0, 0, 0, 0.5)",
            pointerEvents: "none",
          }}
        >
          <strong style={{ color: "#60a5fa", display: "block", marginBottom: "4px" }}>
            Chunk {hoveredNode.id}
          </strong>
          <span style={{ fontSize: "11px", display: "block", fontStyle: "italic" }}>
            {hoveredNode.text.slice(0, 160)}
            {hoveredNode.text.length > 160 ? "..." : ""}
          </span>
        </div>
      )}
    </div>
  );
}
