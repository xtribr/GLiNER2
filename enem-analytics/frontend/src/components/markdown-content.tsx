import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface MarkdownContentProps {
  content: string;
}

export function MarkdownContent({ content }: MarkdownContentProps) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        h1: ({ children }) => (
          <h1 className="mb-2 text-3xl font-black text-slate-950 sm:text-4xl">
            {children}
          </h1>
        ),
        h2: ({ children }) => (
          <h2 className="mb-3 mt-10 text-xl font-bold text-slate-950 first:mt-0">
            {children}
          </h2>
        ),
        h3: ({ children }) => (
          <h3 className="mb-2 mt-6 text-base font-bold text-slate-800">
            {children}
          </h3>
        ),
        p: ({ children }) => (
          <p className="mb-4 text-sm leading-7 text-slate-700">
            {children}
          </p>
        ),
        ul: ({ children }) => (
          <ul className="mb-4 list-disc space-y-1.5 pl-5">
            {children}
          </ul>
        ),
        ol: ({ children }) => (
          <ol className="mb-4 list-decimal space-y-1.5 pl-5">
            {children}
          </ol>
        ),
        li: ({ children }) => (
          <li className="text-sm leading-7 text-slate-700 marker:text-slate-400">
            {children}
          </li>
        ),
        strong: ({ children }) => (
          <strong className="font-semibold text-slate-900">
            {children}
          </strong>
        ),
        a: ({ href, children }) => (
          <a
            href={href}
            className="text-sky-600 underline underline-offset-2 hover:text-sky-800"
            target={href?.startsWith("http") ? "_blank" : undefined}
            rel={href?.startsWith("http") ? "noopener noreferrer" : undefined}
          >
            {children}
          </a>
        ),
        hr: () => <hr className="my-8 border-slate-200" />,
        blockquote: ({ children }) => (
          <blockquote className="my-4 border-l-4 border-slate-200 pl-4 text-sm text-slate-600 italic">
            {children}
          </blockquote>
        ),
        table: ({ children }) => (
          <div className="mb-6 w-full overflow-x-auto rounded-lg border border-slate-200">
            <table className="w-full min-w-170 border-collapse bg-white">
              {children}
            </table>
          </div>
        ),
        thead: ({ children }) => (
          <thead className="bg-slate-50">{children}</thead>
        ),
        tbody: ({ children }) => <tbody>{children}</tbody>,
        tr: ({ children }) => (
          <tr className="border-b border-slate-200 last:border-b-0">
            {children}
          </tr>
        ),
        th: ({ children }) => (
          <th className="px-3 py-2 text-left text-xs font-semibold text-slate-700">
            {children}
          </th>
        ),
        td: ({ children }) => (
          <td className="px-3 py-2 text-sm leading-6 text-slate-700">
            {children}
          </td>
        ),
      }}
    >
      {content}
    </ReactMarkdown>
  );
}
